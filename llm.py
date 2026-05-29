from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

# prepare the model input
class llm:
    def __init__(self, name):

        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto"
        )

        self.history = []
    
    def __call__(self, prompt):
        print("prompt:", prompt)
        self.history.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)
        self.history.append({"role": "assistant", "content": thinking_content + content})
        return content

TEXT = """You are playing a set of Hearts games.
Commands:
1. START list[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
   A game starts. Receive exactly 13 cards in format: (type (0: spades, 1: hearts, 2: clovers, 3: diamonds), [0~12] referring to [2~10, J, Q, K, A]).
   Output format: Any
2. PASS int
   Submit exactly three cards in your card list to pass to (0: left, 1: right, 2: across).
   Output format: list[tuple[int, int], tuple[int, int], tuple[int, int]]
3. GET list[tuple[int, int], tuple[int, int], tuple[int, int]]
   Get the passed cards, and insert them into your card list.
   Output format: Any
4. PLAY list[tuple[int, int], ...]
   It is your turn to play. You will receive the played cards in this round. Decide which card you play in this round, and submit it.
   Output format: tuple[int, int]
5. TRICK list[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
   Remmber the whole trick.
   Output format: Any
Reminder: output in plain text instead of a ```python``` markdown container.
Are you ready?"""

if __name__ == '__main__':
    LLM = llm(os.path.join(os.environ['USERPROFILE'],
                           '.cache/modelscope/hub/models/Qwen/Qwen3-0.6B'))

    LLM(TEXT)
    
    LLM("START [(0, 1), (0, 5), (0, 6), (0, 11), "
        "(1, 6), (1, 7), (1, 11), (1, 12), "
        "(2, 2), (2, 4), (2, 12), "
        "(3, 0), (3, 8)]")

    LLM("PASS 0")

    LLM("GET [(2, 11), (3, 10), (3, 11)]")

    LLM("PLAY [(2, 0)]")