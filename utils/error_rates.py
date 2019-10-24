import editdistance
def cer(r, h):
    #Remove any double or trailing
    r = u' '.join(r.decode('utf-8').split())
    h = u' '.join(h.split())

    return err(r, h)

def err(r, h):
    dis = editdistance.eval(r, h)
    if len(r) == 0.0:
        return len(h)

    return float(dis) / float(len(r))

def wer(r, h):
    r = r.split()
    h = h.split()

    return err(r,h)
