import torch
from transformers import LlamaForCausalLM, AutoTokenizer, pipeline
from langchain import LLMChain

class llama_prompt(torch.Module):
    def __init__(self,
                 model_name,
                 device):
        self.llm = LlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.pipe = self.get_pipe()

    def get_sentence_list(self, data):
        sentence_list = []
        batch_size = data['input_ids'].shape[0]
        for batch_index in range(batch_size):
            real_word_index = data['word_ids'][batch_index] != -100
            sentence = self.tokenizer.decode(data['input_ids'][batch_index][real_word_index])
            sentence_list.append(sentence)

        return sentence_list

    def get_pipe(self):
        pipe = pipeline("text-generation",
                        model=self.llm,
                        tokenizer=self.tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device,
                        max_new_tokens=512,
                        do_sample=True,
                        top_k=30,
                        num_return_sequences=1,
                        eos_token_id=self.tokenizer.eos_token_id
                        )

    def forward(self):
        pass
