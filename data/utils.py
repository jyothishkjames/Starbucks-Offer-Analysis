# Cleaning value
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


# Cleaning Channels
def channel_1(x):
    try:
        value = x[0]

        return value

    except:

        return float("NAN")


def channel_2(x):
    try:
        value = x[1]

        return value

    except:

        return float("NAN")


def channel_3(x):
    try:
        value = x[2]

        return value

    except:

        return float("NAN")


def channel_4(x):
    try:
        value = x[3]

        return value

    except:

        return float("NAN")


