
def offer(x):
    try:

        value = x['offer id']

        return value

    except:

        try:

            value = x['offer_id']

            return value

        except:

            return float("NAN")


def amount(x):
    try:

        value = x['amount']

        return value

    except:

        return float("NAN")
