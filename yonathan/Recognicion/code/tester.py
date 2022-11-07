from models.Attention import Attention
from Configs.Config import Config


def main():
    config = Config()
    attention_model = Attention(config=config)
    attention_path = "/home/idanta/BU-TD/yonathan/Recognicion/code/Configs/Attention.yaml"
    Attention.document2config(
        attention_model, attention_path, replace_now=True)
    model = Attention(config=config)
    print(model)


if __name__ == "__main__":
    main()
