"""
Simulate OCR-only answering on 50 test questions without needing a GPU.
Tests the new direction/between/proximity logic.
"""

import sys
from inference import find_text_on_map, answer_with_ocr, get_spatial_zone
import pandas as pd

# Simulated OCR texts extracted from the stitched map
ocr_data = [
    {'text': 'Powai Lake', 'cx': 550, 'cy': 400, 'norm_x': 0.5, 'norm_y': 0.35, 'confidence': 0.95},
    {'text': 'Vihar Lake', 'cx': 600, 'cy': 200, 'norm_x': 0.55, 'norm_y': 0.15, 'confidence': 0.9},
    {'text': 'Vihar Lake Dam', 'cx': 580, 'cy': 150, 'norm_x': 0.53, 'norm_y': 0.1, 'confidence': 0.85},
    {'text': 'IIT Bombay', 'cx': 700, 'cy': 380, 'norm_x': 0.65, 'norm_y': 0.33, 'confidence': 0.9},
    {'text': 'Indian Institute of Technology Bombay', 'cx': 700, 'cy': 370, 'norm_x': 0.65, 'norm_y': 0.32, 'confidence': 0.8},
    {'text': 'SEEPZ', 'cx': 250, 'cy': 300, 'norm_x': 0.23, 'norm_y': 0.26, 'confidence': 0.95},
    {'text': 'CSMI Airport', 'cx': 150, 'cy': 650, 'norm_x': 0.14, 'norm_y': 0.57, 'confidence': 0.9},
    {'text': 'Airport Road', 'cx': 180, 'cy': 680, 'norm_x': 0.16, 'norm_y': 0.6, 'confidence': 0.85},
    {'text': 'Tunga Village', 'cx': 500, 'cy': 500, 'norm_x': 0.46, 'norm_y': 0.44, 'confidence': 0.8},
    {'text': 'Takshila Colony', 'cx': 200, 'cy': 250, 'norm_x': 0.18, 'norm_y': 0.22, 'confidence': 0.85},
    {'text': 'Sahar Elevated Road', 'cx': 200, 'cy': 620, 'norm_x': 0.18, 'norm_y': 0.54, 'confidence': 0.8},
    {'text': 'Hanuman Tekdi', 'cx': 750, 'cy': 350, 'norm_x': 0.69, 'norm_y': 0.3, 'confidence': 0.75},
    {'text': 'MIDC Andheri', 'cx': 280, 'cy': 450, 'norm_x': 0.26, 'norm_y': 0.39, 'confidence': 0.85},
    {'text': 'Andheri Veravali Reservoir', 'cx': 100, 'cy': 200, 'norm_x': 0.09, 'norm_y': 0.17, 'confidence': 0.75},
    {'text': 'Aarey Metro Depot', 'cx': 350, 'cy': 380, 'norm_x': 0.32, 'norm_y': 0.33, 'confidence': 0.7},
    {'text': 'Mohli Village', 'cx': 450, 'cy': 700, 'norm_x': 0.41, 'norm_y': 0.61, 'confidence': 0.7},
    {'text': 'Narayan Nagar', 'cx': 450, 'cy': 750, 'norm_x': 0.41, 'norm_y': 0.65, 'confidence': 0.8},
    {'text': 'Marol Maroshi Road', 'cx': 350, 'cy': 550, 'norm_x': 0.32, 'norm_y': 0.48, 'confidence': 0.8},
    {'text': 'Cliff Avenue', 'cx': 500, 'cy': 460, 'norm_x': 0.46, 'norm_y': 0.4, 'confidence': 0.7},
    {'text': 'Peru Baug', 'cx': 500, 'cy': 320, 'norm_x': 0.46, 'norm_y': 0.28, 'confidence': 0.7},
    {'text': 'Kherani Road', 'cx': 600, 'cy': 650, 'norm_x': 0.55, 'norm_y': 0.57, 'confidence': 0.75},
    {'text': 'National Institute of Industrial Engineering', 'cx': 650, 'cy': 200, 'norm_x': 0.6, 'norm_y': 0.17, 'confidence': 0.75},
    {'text': 'Kondivita Village', 'cx': 350, 'cy': 500, 'norm_x': 0.32, 'norm_y': 0.44, 'confidence': 0.7},
    {'text': 'Krantiveer Lahuji Salve Marg', 'cx': 250, 'cy': 350, 'norm_x': 0.23, 'norm_y': 0.3, 'confidence': 0.7},
    {'text': 'Hiranandani Gardens', 'cx': 650, 'cy': 400, 'norm_x': 0.6, 'norm_y': 0.35, 'confidence': 0.8},
    {'text': 'Poonam Nagar', 'cx': 100, 'cy': 150, 'norm_x': 0.09, 'norm_y': 0.13, 'confidence': 0.75},
    {'text': 'Adi Shankaracharya Marg', 'cx': 500, 'cy': 450, 'norm_x': 0.46, 'norm_y': 0.39, 'confidence': 0.7},
    {'text': 'Park Site Colony', 'cx': 700, 'cy': 600, 'norm_x': 0.65, 'norm_y': 0.52, 'confidence': 0.7},
    {'text': 'Marol Naka L1', 'cx': 350, 'cy': 570, 'norm_x': 0.32, 'norm_y': 0.5, 'confidence': 0.7},
    {'text': 'Model Town', 'cx': 150, 'cy': 200, 'norm_x': 0.14, 'norm_y': 0.17, 'confidence': 0.7},
    {'text': 'Aarey Colony', 'cx': 400, 'cy': 300, 'norm_x': 0.37, 'norm_y': 0.26, 'confidence': 0.8},
    {'text': 'Mithi River', 'cx': 400, 'cy': 550, 'norm_x': 0.37, 'norm_y': 0.48, 'confidence': 0.7},
    {'text': 'Asalpha', 'cx': 500, 'cy': 750, 'norm_x': 0.46, 'norm_y': 0.65, 'confidence': 0.7},
    {'text': 'Saki Naka', 'cx': 400, 'cy': 650, 'norm_x': 0.37, 'norm_y': 0.57, 'confidence': 0.8},
    {'text': 'Andheri-Kurla Road', 'cx': 350, 'cy': 700, 'norm_x': 0.32, 'norm_y': 0.61, 'confidence': 0.7},
    {'text': 'Jogeshwari Vikhroli Link Road', 'cx': 400, 'cy': 400, 'norm_x': 0.37, 'norm_y': 0.35, 'confidence': 0.65},
    {'text': 'JVLR', 'cx': 400, 'cy': 400, 'norm_x': 0.37, 'norm_y': 0.35, 'confidence': 0.8},
    {'text': 'Technology Street', 'cx': 680, 'cy': 360, 'norm_x': 0.63, 'norm_y': 0.31, 'confidence': 0.65},
    {'text': 'Bhavani Nagar', 'cx': 350, 'cy': 480, 'norm_x': 0.32, 'norm_y': 0.42, 'confidence': 0.7},
    {'text': 'Larsen and Toubro', 'cx': 500, 'cy': 550, 'norm_x': 0.46, 'norm_y': 0.48, 'confidence': 0.7},
    {'text': 'Marol Village', 'cx': 350, 'cy': 550, 'norm_x': 0.32, 'norm_y': 0.48, 'confidence': 0.8},
    {'text': 'Nityanand Nagar', 'cx': 700, 'cy': 800, 'norm_x': 0.65, 'norm_y': 0.7, 'confidence': 0.7},
    {'text': 'Terminal 2', 'cx': 140, 'cy': 660, 'norm_x': 0.13, 'norm_y': 0.58, 'confidence': 0.8},
    {'text': 'Western Express Highway', 'cx': 200, 'cy': 500, 'norm_x': 0.18, 'norm_y': 0.44, 'confidence': 0.7},
    {'text': 'Chandivali', 'cx': 650, 'cy': 550, 'norm_x': 0.6, 'norm_y': 0.48, 'confidence': 0.75},
    {'text': 'Saki Vihar Road', 'cx': 500, 'cy': 500, 'norm_x': 0.46, 'norm_y': 0.44, 'confidence': 0.7},
]

map_h, map_w = 1152, 1088

df = pd.read_csv('test_50_questions.csv')
correct = 0
wrong = 0
deferred = 0

for _, row in df.iterrows():
    qid = row['id']
    question = row['question']
    opts = [row['option_1'], row['option_2'], row['option_3'], row['option_4']]
    gt = row['correct_answer']

    ans, conf = answer_with_ocr(question, opts, ocr_data, map_h, map_w)
    if ans is not None and conf >= 0.6:
        if ans == gt:
            correct += 1
            status = 'OK'
        else:
            wrong += 1
            status = 'WRONG'
        print(f'{qid}: OCR={ans} GT={gt} conf={conf:.2f} {status}  | {question[:60]}')
    else:
        deferred += 1
        print(f'{qid}: DEFER (ans={ans}, conf={conf:.2f})  | {question[:60]}')

print(f'\n{"="*60}')
print(f'Correct: {correct}, Wrong: {wrong}, Deferred to VLM: {deferred}')
print(f'OCR score: {correct - 0.25*wrong:.2f}')
print(f'Remaining for VLM: {deferred} questions (estimate 80-90% VLM accuracy -> +{int(deferred * 0.85):.0f} correct)')
print(f'Combined estimate: {correct + int(deferred * 0.85):.0f}/50')
print(f'{"="*60}')
