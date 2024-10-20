def time_str(t):
    if t < 60:
        t_str = '%.3fs' % t
    elif t < 3600:
        t_m = t // 60
        t_s = t - t_m * 60
        t_str = '%.0fm ' % t_m
        if t_s > 0:
            t_str = t_str + '%.3fs' % t_s
        # t_str = '%.0fm %.3fs' % (t_m, t_s)
    elif t < 86400:
        t_h = t // 3600
        temp = t - t_h * 3600
        t_m = temp // 60
        t_s = temp - t_m * 60
        t_str = '%.0fh ' % t_h
        if t_m > 0:
            t_str = t_str + '%.0fm ' % t_m
        if t_s > 0:
            t_str = t_str + '%.3fs' % t_s
    else:
        t_d = t // 86400
        temp = t - t_d * 86400
        t_h = temp // 3600
        temp = temp - t_h * 3600
        t_m = temp // 60
        t_s = temp - t_m * 60
        t_str = '%.0fd ' % t_d
        if t_h > 0:
            t_str = t_str + '%.0fh ' % t_h
        if t_m > 0:
            t_str = t_str + '%.0fm ' % t_m
        if t_s > 0:
            t_str = t_str + '%.3fs' % t_s
    return t_str

def early_stop_check(loss_all, loss_last100, i1):
    early_stop=0
    loss_current100=loss_last100
    if i1%100==0:
        loss_current100=sum(loss_all[i1-99:i1+1])
        if loss_current100>=loss_last100:
            early_stop=1

    return early_stop, loss_current100
    