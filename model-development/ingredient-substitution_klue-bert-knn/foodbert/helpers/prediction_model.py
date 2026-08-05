import json

import torch
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer

from foodbert.helpers.instructions_dataset import InstructionsDataset


class PredictionModel:

    def __init__(self):
        self.model: BertModel = BertModel.from_pretrained(
            pretrained_model_name_or_path='foodbert/data/mlm_output_v2')
        with open('foodbert/data/used_ingredients.json', 'r', encoding='utf-8') as f:
            used_ingredients = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained('foodbert/data/mlm_output_v2', never_split=used_ingredients)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def predict_embeddings(self, sentences):
        dataset = InstructionsDataset(tokenizer=self.tokenizer, sentences=sentences)
        dataloader = DataLoader(dataset, batch_size=100, pin_memory=True)

        embeddings = []
        input_ids_list = []
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                embeddings_batch = self.model(batch)
                embeddings.extend(embeddings_batch[0])
                input_ids_list.extend(batch)

        return torch.stack(embeddings), input_ids_list

    def get_ingredient_embedding(self, sentence, ingredient_name):
        """재료 이름의 서브워드 토큰 임베딩을 평균내어 반환"""
        encoding = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128)
        input_ids = encoding['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 재료 이름의 서브워드 토큰 찾기
        ingredient_tokens = self.tokenizer.tokenize(ingredient_name)

        # 문장 내에서 재료 토큰 위치 찾기
        positions = []
        for i in range(len(tokens) - len(ingredient_tokens) + 1):
            if tokens[i:i+len(ingredient_tokens)] == ingredient_tokens:
                positions.extend(range(i, i+len(ingredient_tokens)))
                break

        if not positions:
            return None

        with torch.no_grad():
            input_ids_tensor = input_ids.unsqueeze(0).to(self.device)
            output = self.model(input_ids_tensor)
            hidden_states = output[0][0]  # (seq_len, 768)

        # 해당 위치 임베딩들의 평균
        ingredient_embedding = hidden_states[positions].mean(dim=0).cpu().numpy()
        return ingredient_embedding
