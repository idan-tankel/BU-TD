from models.Attention import Attention
from Configs.Config import Config


def main():
    print("hello world")
    config = Config()
    attention_model = Attention(config=config)
    attention_path = "/home/idanta/BU-TD/yonathan/Recognicion/code/Configs/Attention.yaml"
    Attention.document2attentionconfig(
        attention_model, attention_path, override=True)
    model = Attention(config=config)
    print(model)


if __name__ == "__main__":
    main()
