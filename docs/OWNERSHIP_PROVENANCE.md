# Original weights provenance policy

The model is **initialized from random tensors**. No pretrained checkpoint is used in the source code. All original weights trained here should be identified by run ID, sha256 of dataset and code, training configuration, tokenizer version, and training log. The bundled demonstration corpus was authored for engineering tests. Future datasets must be licensed/consented and checked for leaks; using foreign model outputs is not automatically allowed. Third-party framework licenses still apply. No legal guarantee of exclusive copyright in machine-generated model weights is claimed.

The checkpoint also contains an optimizer state; it is not a complete artifact if the dataset snapshot and all model/tokenizer software versions are missing. Back up those separately before decommissioning any server.
