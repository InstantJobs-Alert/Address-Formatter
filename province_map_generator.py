import random
import sys
import os, subprocess, time
from collections import defaultdict
import time


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("matplotlib")
install("ollama")

import matplotlib.pyplot as plt
import ollama


def wrap_prompt(preprompt: str, input: str) -> str:
    return preprompt + "\n" + input


class DefaultProvinceMap:
    def __init__(self):
        self.map = {
            "alabama": "AL",
            "alaska": "AK",
            "arizona": "AZ",
            "arkansas": "AR",
            "california": "CA",
            "colorado": "CO",
            "connecticut": "CT",
            "delaware": "DE",
            "florida": "FL",
            "georgia": "GA",
            "hawaii": "HI",
            "idaho": "ID",
            "illinois": "IL",
            "indiana": "IN",
            "iowa": "IA",
            "kansas": "KS",
            "kentucky": "KY",
            "louisiana": "LA",
            "maine": "ME",
            "maryland": "MD",
            "massachusetts": "MA",
            "michigan": "MI",
            "minnesota": "MN",
            "mississippi": "MS",
            "missouri": "MO",
            "montana": "MT",
            "nebraska": "NE",
            "nevada": "NV",
            "new hampshire": "NH",
            "new jersey": "NJ",
            "new mexico": "NM",
            "new york": "NY",
            "north carolina": "NC",
            "north dakota": "ND",
            "ohio": "OH",
            "oklahoma": "OK",
            "oregon": "OR",
            "pennsylvania": "PA",
            "rhode island": "RI",
            "south carolina": "SC",
            "south dakota": "SD",
            "tennessee": "TN",
            "texas": "TX",
            "utah": "UT",
            "vermont": "VT",
            "virginia": "VA",
            "washington": "WA",
            "west virginia": "WV",
            "wisconsin": "WI",
            "wyoming": "WY",
            "district of columbia": "DC",
            "washington dc": "DC",
            "alberta": "AB",
            "british columbia": "BC",
            "manitoba": "MB",
            "new brunswick": "NB",
            "newfoundland and labrador": "NL",
            "nova scotia": "NS",
            "northwest territories": "NT",
            "nunavut": "NU",
            "ontario": "ON",
            "prince edward island": "PE",
            "quebec": "QC",
            "saskatchewan": "SK",
            "yukon": "YT",
            "al": "AL",
            "ak": "AK",
            "az": "AZ",
            "ar": "AR",
            "ca": "CA",
            "co": "CO",
            "ct": "CT",
            "de": "DE",
            "fl": "FL",
            "ga": "GA",
            "hi": "HI",
            "id": "ID",
            "il": "IL",
            "in": "IN",
            "ia": "IA",
            "ks": "KS",
            "ky": "KY",
            "la": "LA",
            "me": "ME",
            "md": "MD",
            "ma": "MA",
            "mi": "MI",
            "mn": "MN",
            "ms": "MS",
            "mo": "MO",
            "mt": "MT",
            "ne": "NE",
            "nv": "NV",
            "nh": "NH",
            "nj": "NJ",
            "nm": "NM",
            "ny": "NY",
            "nc": "NC",
            "nd": "ND",
            "oh": "OH",
            "ok": "OK",
            "or": "OR",
            "pa": "PA",
            "ri": "RI",
            "sc": "SC",
            "sd": "SD",
            "tn": "TN",
            "tx": "TX",
            "ut": "UT",
            "vt": "VT",
            "va": "VA",
            "wa": "WA",
            "wv": "WV",
            "wi": "WI",
            "wy": "WY",
            "dc": "DC",
            "ab": "AB",
            "bc": "BC",
            "mb": "MB",
            "nb": "NB",
            "nl": "NL",
            "ns": "NS",
            "nt": "NT",
            "nu": "NU",
            "on": "ON",
            "pe": "PE",
            "qc": "QC",
            "sk": "SK",
            "yt": "YT",
        }

    def check_map(self):
        if len(self.map) == 129:
            print("-----------map is complete!-----------")
            return True
        else:
            raise Exception("Map is not complete. Please check the map.")

    def get_default_map(self):
        try:
            print("-----------checking map-----------")
            self.check_map()
            return self.map
        except Exception as e:
            print(f"Error: {e}")
            return None


class LLMInvoke:
    def __init__(self, model: str = None):
        self.selected_model = model

    def set_model(self, model: str):
        self.selected_model = model

    def invoke(self, input: str) -> str:
        if not self.selected_model:
            return "No model selected."
        try:
            response = ollama.chat(
                model=self.selected_model,
                messages=[
                    {
                        "role": "user",
                        "temperature": 0,
                        "repeat_penalty": 6,
                        "content": input,
                        "num_ctx": 4096,
                    },
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            return str(e)


def map_data_analysis(
        model_name, certain_count, uncertain_count, illegal_count, runtime
):
    categories = ["Certain", "Uncertain", "Illegal"]
    counts = [certain_count, uncertain_count, illegal_count]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color=["green", "orange", "red"])
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha="center", va="bottom", fontsize=12)
    plt.title(f"{model_name} (Time: {runtime:.2f} s)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.ylim(0, max(counts) + 50)

    # Save the figure to avoid overwriting
    plt.savefig(f"{time.time()}_analysis.png")


def build_province_map(model_name="llama3.1:8b", iter=10):
    print('\n\n')
    print(
        f"-------------------Building province map using {model_name}-------------------"
    )
    time.sleep(3)
    start_time = time.time()

    province_map = defaultdict(list)
    illegal_provinces = set()

    with open("unique_provinces.txt", "r") as file:
        lines = file.readlines()
        total_line = len(lines) - 1

    for i in range(iter):
        progress = 0
        print('\n')
        for line in lines:
            line = line.replace('"', "")
            line = line.replace(",", " ")
            key = line.strip().lower()
            word_split = key.split()

            if len(word_split) == 0:
                #print(key, "empty key")
                continue
            else:
                first_word = word_split[0]
                last_word = word_split[-1]

            progress += 1

            time.sleep(0.001)

            sys.stdout.write(
                f"\rCurrent Iteration: {i + 1}, Model Name: {model_name}, Progress: {progress}/{total_line}")
            sys.stdout.flush()

            if key in default_province_map:
                province_map[key].append(default_province_map[key])
                continue

            if (
                    first_word
                    and last_word
                    and first_word in default_province_map
                    and last_word in default_province_map
            ):
                illegal_provinces.add(key)
                continue

            if first_word and first_word in default_province_map:
                province_map[key].append(default_province_map[first_word])
                continue

            if last_word and last_word in default_province_map:
                province_map[key].append(default_province_map[last_word])
                continue

            province_map_full_prompt = wrap_prompt(province_map_prompt, key)
            value = llm_invoke.invoke(province_map_full_prompt)
            value = value.strip()

            if (
                    value
                    and value != '""'
                    and value != "None"
                    and value != "''"
                    and len(value) <= 4
                    and key not in illegal_provinces
            ):
                # print(f"Key: {key}, Value: {value}")
                province_map[key].append(value)
            else:
                # print("miss! ", key)
                illegal_provinces.add(key)

    certain_cnt = 0
    uncertain_cnt = 0
    illegal_cnt = 0

    # use random number to avoid overwriting
    random_number = random.randint(0, 1000)

    with open(f"certain_province_map_{random_number}.txt", "w") as certain_file:
        with open(f"uncertain_province_map_{random_number}.txt", "w") as uncertain_file:
            with open(f"illegal_provinces.txt_{random_number}", "w") as illegal_file:
                for key, value in province_map.items():
                    if len(set(value)) == 1:
                        certain_file.write(f'"{key}": "{value[0]}"\n')
                        certain_cnt += 1
                    else:
                        uncertain_file.write(f'"{key}": "{value}"\n')
                        uncertain_cnt += 1

                for illegal_province in illegal_provinces:
                    illegal_file.write(illegal_province + "\n")
                    illegal_cnt += 1

    runtime = time.time() - start_time
    return certain_cnt, uncertain_cnt, illegal_cnt, runtime, random_number


def find_common_lines(file1, file2, file3):
    with open(file1, "r") as f1:
        lines1 = f1.readlines()

    with open(file2, "r") as f2:
        lines2 = f2.readlines()

    with open(file3, "r") as f3:
        lines3 = f3.readlines()

    common_lines = set(lines1) & set(lines2) & set(lines3)

    with open("common_lines.txt", "w") as f:
        for line in common_lines:
            f.write(line)

    return common_lines


def generate_go_map():
    with open("provinceMap.go", "w") as go_file:
        go_file.write("package utils\n\n")
        go_file.write("var ProvinceMap = map[string]string{\n")

        with open("common_lines.txt", "r") as f:
            lines = f.readlines()

            for line in lines:
                key, value = line.strip().split(":")
                go_file.write("\t")
                go_file.write(f"{key}: {value},\n")

            go_file.write("}\n")


print("----------------Welcome to the province map generator!----------------")

model_list = [
    "llama3.1:8b",
    "mistral-nemo:latest",
    "mistral:7b",
]

test_input = "where is Stanford Univ? just give me the city without explanation"

llm_invoke = LLMInvoke()

print("----------------Loading Models and Testing Models!----------------")
print(test_input)
for model in model_list:
    llm_invoke.set_model(model)
    print(f"Model: {model}")
    print(llm_invoke.invoke(test_input))
    print("\n\n")

province_map_prompt = """
Given is a geo info about an address. You should define this data and output An Abbreviation of province/state that is related to the location. You should ONLY output US/CANADA province/states.

i.e. 
Miami will output FL
19104 will output PA
Texas will output TX
Ontario will output ON

If the input is a street/city without a postcode and you cannot 100% define the provice, output empty string "".
ONLY output province name in plain text, NO explaination, NO code. 

Your input:


"""

default_province_map = DefaultProvinceMap().get_default_map()

random_numbers = []

for idx, model in enumerate(model_list):
    llm_invoke.set_model(model)
    certain_count, uncertain_count, illegal_count, runtime, new_random = build_province_map(model_name=model, iter=3)
    map_data_analysis(model, certain_count, uncertain_count, illegal_count, runtime)
    random_numbers.append(new_random)

print("\n\n-------------------Generating Go Map-------------------\n\n")
file1 = f"certain_province_map_{random_numbers[0]}.txt"
file2 = f"certain_province_map_{random_numbers[1]}.txt"
file3 = f"certain_province_map_{random_numbers[2]}.txt"

print("-------------------Congratulations! Your Map is Generated!-------------------")

find_common_lines(file1, file2, file3)

generate_go_map()
