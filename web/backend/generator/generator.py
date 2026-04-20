from transformers import AutoModelForCausalLM



class MusAcGen:
    def __init__(self):
        AMT_model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-medium-800k').cuda()
    def generate(self, file_location):
        relative_fl = "../"+file_location
        
        
