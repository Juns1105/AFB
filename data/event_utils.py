"""Event-to-frame conversion.

Adapted from event_utils: https://github.com/TimoStoff/event_utils
"""
import torch


def binary_search_torch_tensor(t, l, r, x, side='left'):
    """Return the index of the event nearest to timestamp `x` in the sorted tensor `t`."""
    if r is None:
        r = len(t) - 1
    while l <= r:
        mid = l + (r - l) // 2
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


def events_to_image(xs, ys, ps, sensor_size):
    """Accumulate event polarities per pixel."""
    img = torch.zeros(sensor_size)
    img.index_put_((ys.long(), xs.long()), ps, accumulate=True)
    return img


def events_to_stack(xs, ys, ts, ps, num_bins, sensor_size):
    """Split events into `num_bins` equal time slices and accumulate each one: E_N in (N, H, W)."""
    dt = (ts[-1] - ts[0]) / num_bins
    bins = []
    for bi in range(num_bins):
        tstart = ts[0] + dt * bi
        tend = tstart + dt
        beg = binary_search_torch_tensor(ts, 0, len(ts) - 1, tstart)
        end = binary_search_torch_tensor(ts, 0, len(ts) - 1, tend)
        bins.append(events_to_image(xs[beg:end], ys[beg:end], ps[beg:end], sensor_size))
    return torch.stack(bins)
