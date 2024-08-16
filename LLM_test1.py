import random
import sys
import os, subprocess, time
from collections import defaultdict
import time
from llm_invoke import LLMInvoke
from default_address_map import DefaultAddressMap
from provinceMap import province_super_map


def wrap_prompt(pre_prompt: str, current_input: str) -> str:
    return pre_prompt + "\n" + current_input


def extract_country(adr_line: str) -> str:
    country_prompt = """
    Where is this address from? ONLY output one country in plain text, no explanation, no code. Make sure your answer is in the correct format.
    If you are not 100% sure, please output an empty string "".

    Make sure your answer is in the correct

    Your input:
    """
    split_line = adr_line.split(',')
    country_entry = split_line[0].strip().lower()

    if country_entry in default_country_map:
        country_entry = default_country_map[country_entry]
        # test_results.write(f"{adr_line} : {country_entry}\n")
    else:
        parsed_address = ""
        for part in split_line:
            part = part.strip()
            if part != '\"\"':
                parsed_address += part + ", "

        full_prompt = wrap_prompt(country_prompt, parsed_address)
        response = llm_invoke.invoke(full_prompt).strip().lower()
        if response in default_country_map:
            country_entry = default_country_map[response]
            # test_results.write(f"{adr_line} : {country_entry}\n")
        else:
            #print(response)
            country_entry = "invalid"

    return country_entry


def extract_province(adr_line: str, current_country: str) -> str:
    province_prompt = """
        Which province/state is the following address located in? You should output the province name of this address in plain text.
        If you are not 100% sure, please output an empty string "".
        ONLY output province name in plain text, NO EXPLANATION, NO code.
        
        Your input:
        
        """

    province_prompt_with_valid_country = f"""
        Which province/state is the following address located in the {current_country}? You should output the province name of this address in plain text.
        If you are not 100% sure, please output an empty string "".
        ONLY output province name in plain text, NO EXPLANATION, NO code.

        Your input:
    """

    second_attempt_prompt = """
        Find the expected province/state name of given response, ONLY output province name in plain text, NO EXPLANATION, NO code.
        If input is "" or none, please output 'Invalid'.
    """

    split_line = adr_line.split(',')
    province_entry = split_line[1].strip().lower()

    # use a list to store the provinces results, return the most common one
    provinces = []

    # case 1: it can be found in the large province map
    if province_entry in province_super_map:
        province_entry = province_super_map[province_entry]
        # test_results.write(f"{adr_line} : {province_entry}\n")
        return province_entry

    # now we need to reformat the address:
    parsed_address = ""
    for part in split_line:
        part = part.strip()
        if part != '\"\"':
            parsed_address += part + ", "

    if current_country != "":
        parsed_address = parsed_address + current_country
    else:
        parsed_address = parsed_address[:-2]

    for i in range(5):
        # case 2: Use LLM to predict the province
        # if country is valid, use country to help predict province
        if len(current_country) > 1:
            full_prompt = wrap_prompt(province_prompt_with_valid_country, parsed_address)
        else:
            full_prompt = wrap_prompt(province_prompt, parsed_address)

        response = llm_invoke.invoke(full_prompt).strip().lower()

        if response in province_super_map:
            province_entry = province_super_map[response]
            # test_results.write(f"{adr_line} : {province_entry}\n")
        elif len(response) > 10:
            # print("No! An EXPLANATION is found for" + parsed_address)
            # if output is an explanation, try to parse the explanation
            second_attempt_full_prompt = wrap_prompt(second_attempt_prompt, response)
            response = llm_invoke.invoke(second_attempt_full_prompt).strip().lower()
            if response in province_super_map:
                province_entry = province_super_map[response]
            else:
                response = llm_invoke.invoke(full_prompt).strip().lower()
                province_entry = "invalid"
        else:
            # TODO: try again with different prompt
            response = llm_invoke.invoke(full_prompt).strip().lower()

        provinces.append(province_entry)

    # return the most common province
    return max(set(provinces), key=provinces.count)


def extract_city(adr_line: str) -> str:
    city_prompt = """
        Find the city name of the following address. You should output the city name in plain text without any explanation. If no city name is found, please output an empty string "".
        ONLY output city name in plain text or "", NO EXPLANATION, NO code.
    
        Your input:
    """

    city_check_prompt = """
        Return 1 if the city name is valid, 0 if the city name is invalid without any explanation.
        ONLY OUTPUT 1 or 0.
        
        Your input:
    """

    remote_check = """
        Given the following address, please determine if it indicates a remote work position (e.g., "work from home" or "remote"): If yes, please respond with "Yes"; if not, please respond with "No" Base your determination on the keywords present in the address.
        ONLY output "Yes", "No" without any explanation.
        
        your input:
    """

    split_line = adr_line.split(',')

    parsed_address = ""
    for part in split_line:
        part = part.strip()
        if part != '\"\"':
            parsed_address += part + ", "
    parsed_address = parsed_address[:-2]

    cities = []
    remotes = []
    for i in range(5):
        city_full_prompt = wrap_prompt(city_prompt, parsed_address)
        remote_full_prompt = wrap_prompt(remote_check, parsed_address)
        response = llm_invoke.invoke(city_full_prompt).strip().lower()
        remote_res = llm_invoke.invoke(remote_full_prompt).strip().lower()

        if remote_res == "yes" or remote_res == "no" or remote_res == "hybrid":
            remotes.append(remote_res)
        else:
            remotes.append("unknown")

        if len(response) > 20:
            # an explanation is found
            response = llm_invoke.invoke(city_full_prompt).strip().lower()

        city_check_full_prompt = wrap_prompt(city_check_prompt, response)
        is_valid = llm_invoke.invoke(city_check_full_prompt).strip()

        if is_valid == "1":
            cities.append(response)
        else:
            response = llm_invoke.invoke(city_full_prompt).strip().lower()
            city_check_full_prompt = wrap_prompt(city_check_prompt, response)
            is_valid = llm_invoke.invoke(city_check_full_prompt).strip()
            if is_valid == "1":
                cities.append(response)

    city_result = max(set(cities), key=cities.count) if cities else ""
    remote_result = max(set(remotes), key=remotes.count)

    return city_result, remote_result


llm_invoke = LLMInvoke("llama3.1:8b")

model_list = [
    'mistral-nemo:latest',
    'gemma2:2b'
    "llama3.1:8b",
]

default_address_maps = DefaultAddressMap()
default_country_map = default_address_maps.get_default_country_map()
default_province_map = default_address_maps.get_default_province_map()
# santa_success = 0
# ny_success = 0
# vc_success = 0
# to_success = 0
# second_hit = 0
# total = 100

for i in range(1):
    # print(f"Test {i + 1} of {total}")
    # run 1000 tests
    with open('test_results.txt', 'w') as test_results:

        with open('address_test.txt', 'r') as address_test:
            # read lines
            lines = address_test.readlines()
            total = len(lines)
            index = 0
            # preprocess lines
            for line in lines:
                if line == '\n' or line == '':
                    continue

                index += 1
                progress_message = f"Processing line {index + 1}/{total}\r"
                sys.stdout.write(progress_message)
                sys.stdout.flush()

                # print('current line:', line)

                # preprocess line
                parts = line.split(',')[1:]
                merged_address = ', '.join([part.strip() for part in parts if part.strip()])

                wrapped_address = {
                    "country": '',
                    "province": '',
                    "city": 'city',
                    "isRemote": ''
                }

                # get country
                wrapped_address['country'] = extract_country(merged_address).strip()

                # get province
                wrapped_address['province'] = extract_province(merged_address, wrapped_address['country']).strip()

                # get city
                city, isRemote = extract_city(merged_address)

                wrapped_address['city'] = city
                wrapped_address['isRemote'] = isRemote

                test_results.write(f"{line} : {wrapped_address}\n")

