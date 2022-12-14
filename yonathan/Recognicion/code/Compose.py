import torch


def compute_pred(batch, model):
    pred = model(batch)
    pred = pred.argmax(dim=1)
    return pred

def Create_new_flag(parser, batch, pred, direction):
    bs = pred.size(0)
    task_type_ohe = torch.nn.functional.one_hot(torch.zeros(bs), 1)
    # Getting the direction embedding, telling which direction we solve now.
    direction_type_ohe = torch.nn.functional.one_hot(direction, parser.ndirections)
    # Getting the character embedding, which character we query about.
    char_type_one = torch.nn.functional.one_hot(pred, 47)
    # Concatenating all three flags into one flag.
    flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=0).float()
    return flag

def compose_tasks(parser, batch,model,directions):
    preds = []
    pred = None
    for direction in directions:
        new_flag = Create_new_flag(parser, batch, pred,direction)
        batch.flag = new_flag
        pred = compute_pred(model, batch)
        preds.append(pred)
        for i in range(pred.size(0)):
            if pred[i] == 47:
             pred[i] = 0

    for i in range(pred.size(0)):
        for prediction in preds:
            if prediction[i] == 47:
                pred[i] = 47
                break
    return pred

