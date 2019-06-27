def unique_key(dp):
    return (dp['linedir'], dp['date'], dp['gtfsTripID'], dp['sequence'])

def unique_key_of_pre(dp):
    return (dp['linedir'], dp['date'], dp['preTripID'], dp['sequence'])

def unique_key_of_next(dp):
    return (dp['linedir'], dp['date'], dp['nextTripID'], dp['sequence'])
