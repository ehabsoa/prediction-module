from flask import Flask, request
import dbio
import htmTraining
import htmPrediction
import json

app = Flask(__name__)

@app.route("/")
def checkrunning():
    return "Delay Prediction Server is running!"


@app.route("/training")
def training():
    linedir = request.args.get("linedir", "1-0")
    from_date = request.args.get("from_date", "00000000")
    to_date = request.args.get("to_date", "30001231")
    htmTraining.trainModel(linedir, from_date, to_date)
    return "Model for linedir {} is (re-)trained.".format(linedir)

@app.route("/prediction")
def prediction():
    linedir = request.args.get("linedir", "1-0")
    date = request.args.get("date", "")
    pr_time = request.args.get("time", "")
    if date == "":
        import datetime
        now = datetime.datetime.now()
        date = now.strftime("%Y%m%d")
    if pr_time == "":
        import datetime
        now = datetime.datetime.now()
        time = now.hour * 3600 + now.minute * 60 + now.second
    else:
        time = int(pr_time)

    cluster, duration = htmPrediction.predictModel(linedir, date, time)
    result_dict = {'cluster':str(cluster), 'duration':str(duration)}
    return json.dumps(result_dict)


def __startup_activities__(run):
    if run:
        print("Getting distinct linedirs.")
        linedirs = dbio.getDistinctLinedirs()
        for ld_tuple in linedirs:
            linedir = ld_tuple[0]
            print("Training for linedir: {}".format(linedir))
            predictAfterDate='20150332'
            htmTraining.trainModel(linedir, to_date=predictAfterDate)


if __name__ == '__main__':
    __startup_activities__(True)
    app.run(host='0.0.0.0')
    print("Server ready.")
