from datetime import datetime, timedelta

def daterange(start_date, end_date, freq="H"):
    if freq == "H":
        delta = timedelta(hours=1)
    else:
        raise Exception("unknown date frequencty type!")
    while start_date < end_date:
        yield start_date
        start_date += delta