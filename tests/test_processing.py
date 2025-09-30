import glob
import datetime
import csv


def LoadProjects(path, projects=[]):
    for filename in glob.glob(path):
        with open(filename, "r", encoding="utf-8") as f:
            fields = [
                "project_name",
                "company_id",
                "employee_id",
                "date_start",
                "date_end",
            ]
            reader = csv.DictReader(f, fields, delimiter=";")
            for row in reader:
                if len(row["project_name"]) < 5:
                    continue
                try:
                    id = int(row["company_id"])
                except:
                    continue
                try:
                    val = int(row["employee_id"])
                except:
                    continue
                try:
                    date_start = datetime.datetime.strptime(
                        row["date_start"], "%Y-%m-%d"
                    )
                except:
                    continue
                try:
                    date_end = datetime.datetime.strptime(
                        row["date_end"], "%Y-%m-%d"
                    )
                except:
                    continue

                Project = {
                    "project_name": row["project_name"],
                    "company_id": id,
                    "employee_id": val,
                    "date_start": date_start,
                    "date_end": date_end,
                }
                projects.append(Project)
    return projects
