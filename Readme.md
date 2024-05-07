
# LTE tasks
sources: 

https://github.com/karpathy/nanoGPT

https://github.com/danielgrittner/nanoGPT-LoRA
## Initial setup (also see source repo):
1) run python3 data/shakespeare_char/prepare.py
2) python3 train_gpt_lte.py

# GaLore tasks
Our GaLore experiment were done by editing/referencing this repo 
https://github.com/jiaweizzhao/GaLore

# LoRA Vision
data used to train/finetune: https://medmnist.com/

meloravit.py - A ViT model with LoRA adapters, mainly using MeLo implementation (GaLore optimizer also used)
https://github.com/JamesQFreeman/LoRA-ViT

ltegalorevit.py - Combined version of lte and galore that we got mostly working, but not enough to fully evaluate. Just needed to figure out which layers to pass in as replica layers in prepare_for_lte
