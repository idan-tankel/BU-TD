The images are batched, so when loading image from the dataset,
the keys are


dict_keys(['
verb # shape (BatchSize,) `Tensor`. The verb IDS - not yet embeddings, just ground truth id's
image # shape (BatchSize,channel,H,W), Tensor
frame_length # len BatchSize
roles
im_name # len BatchSize - the names of the image files loaded
])