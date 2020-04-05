


import re
import datetime
import pandas as pd

def load_data_germany():

    data_path = "data/wikipedia-data.csv"

    reported_cases = list()
    reported_deaths = list()

    with open(data_path, "r") as csv_file:
        for rownum, line in enumerate(csv_file):

            if rownum == 0:
                reported_cases.append(line)
                reported_deaths.append(line)
                continue
            line = line.replace("\n", "")

            death_values = list()
            case_values = list()
            for index, val in enumerate(line.split(";")):
                if index == 0:
                    dval = datetime.datetime.strptime("2020 %s" % val, '%Y %d %b')
                    death_values.append(dval.strftime("%Y.%m.%d"))
                    case_values.append(dval.strftime("%Y.%m.%d"))
                    continue

                if "(" not in line:
                    case_values.append(val)
                    death_values.append(val)
                    continue

                # cases are plain numbers
                case_values.append(re.sub("\(\d+\)", "", val))

                # deaths are in brackets
                death_number = re.findall("\((\d+)\)", val)
                # if death_number:
                #    deaths.append(death_number)
                if death_number:
                    death_values.append(death_number[0])
                else:
                    death_values.append("0")
            reported_cases.append(";".join(case_values) + "\n")
            reported_deaths.append(";".join(death_values) + "\n")

    with open("data/reported_cases.csv", "w+") as csv_file:
        csv_file.writelines(reported_cases)

    with open("data/reported_deaths.csv", "w+") as csv_file:
        csv_file.writelines(reported_deaths)


    cases = pd.read_csv("data/reported_cases.csv", sep=";")
    deaths = pd.read_csv("data/reported_deaths.csv", sep=";")

    cases["time"] = pd.to_datetime(cases.Date, format=format('%Y.%m.%d'), errors="raise")

    cases["T"] = cases.time.apply(func=lambda x: (x - cases.time.min()).days)

    cases["cases"] = cases["Total infections"]
    cases["deaths"] = cases["Total deaths"]

    data = cases[["T", "cases", "deaths"]]
    return data