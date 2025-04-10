def get_incident_types():
    file = open("dataset/incident_types.txt", "r")
    lines = file.readlines()
    file.close

    incident_types = [line.replace("\n", "") for line in lines]
    return list(set(sorted(incident_types)))