from src import gpt2_config
import torch
import torch.nn.functional as F


def generate_article(text, num_samples=1, length=50, temperature=1.2):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(device)

    gpt2_config.GPT2_MODEL.load_state_dict(torch.load("src/model.pt", map_location=torch.device('cpu')), strict=False)

    gpt2_config.GPT2_MODEL.eval()
    gpt2_config.GPT2_MODEL.to(device)

    top_p = 0.9
    top_k = 20

    generated_list=[]

    with torch.no_grad():
        for jj in range(num_samples):
            sum_eval_loss = 0.0
            input_ids = torch.tensor(gpt2_config.GPT2_TOKENIZER.encode(text, add_special_tokens=True)).unsqueeze(0).to(device)
            for _ in range(length):
                outputs = gpt2_config.GPT2_MODEL(input_ids, labels=input_ids)

                loss_eval, logits = outputs[:2]
                sum_eval_loss = sum_eval_loss + loss_eval.item()
                next_token_logits = outputs[1][:, -1, :] / (temperature if temperature > 0 else 1.)
                filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p, top_k=top_k)
                next_token = torch.multinomial(F.softmax(filtered_next_token_logits, dim=-1), num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)

            out = input_ids

            out = out[:, len(input_ids)-1:].tolist()
            for o in out:
                output_text = gpt2_config.GPT2_TOKENIZER.decode(o, clean_up_tokenization_spaces=True)
                generated_list.append(output_text)
                print(output_text)
            eval_mean_loss = sum_eval_loss / length
            eval_perplexity = torch.exp(torch.as_tensor(eval_mean_loss))
            print(f"mean eval loss: {eval_mean_loss}")
            print(f"eval perplexity: {eval_perplexity}")
            print(f"{'-' * 100}")

    return generated_list


# from the gist mentioned in the link
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317#file-top-k-top-p-py
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p(nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits