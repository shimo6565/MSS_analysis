import os
import re
import csv

from dic import smi_to_label 

def make_prompt(system,user,assistant):
    template={"messages":
            [{"role": "system", "content": system},
             {"role": "user", "content": user},
             {"role": "assistant", "content": assistant}]}
    return template

def make_dataset_MSS_to_mol(folder,train_dataset,vali_dataset,system,user,vali_mol):
    for filename in sorted(os.listdir(folder)):#####QA_text内
        if ".DS" in filename:
            continue
        filepath = os.path.join(folder, filename)
        for txt_name in sorted(os.listdir(filepath)):#####  各チャンネルのフォルダ内
            if ".DS" in txt_name:
                continue
            stem = os.path.splitext(os.path.basename(txt_name))[0]
            name = re.sub(r"\d+%?$", "", stem)
            
            txt_path=os.path.join(filepath, txt_name)
            with open(txt_path,'r',encoding='utf-8') as f:
                line = f.readline().strip() 
                devide1=line.split("'concentration'")
                next=devide1[0].split("'molecule'")[1]
                next="'molecule'"+next
                next=next.split("],")[0]+"]"
                next2=user+"'concentration'"+devide1[1].split("}")[0]
                prompt1=make_prompt(system,next2,next)
            if name in vali_mol:
                vali_dataset.append(prompt1)
            else:
                train_dataset.append(prompt1)
    return

def make_dataset_mol_to_odor(input_file,train_dataset,vali_dataset,system,user,vali_mol):
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)#headerをスキップ
        rows = list(reader)
    for row in rows:
        smiles=row[3]
        odor=row[4]
        if smiles in vali_mol:
            prompt = make_prompt(system, user+smi_to_label[smiles], odor)
            vali_dataset.append(prompt)
        else:
            prompt = make_prompt(system, user+smi_to_label[smiles], odor)
            train_dataset.append(prompt)
    return

