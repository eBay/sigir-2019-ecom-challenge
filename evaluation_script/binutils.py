import math

class ExpBins:
    """Divides a range into exponentially sized bins."""
    def __init__(self, range_start, range_end, num_bins, exp_base=math.exp(1), invert=False):
        assert range_start < range_end
        assert (isinstance(num_bins, int) and num_bins > 0)
        assert (exp_base > 0 or exp_base == -1)

        self.range_start = range_start
        self.range_end = range_end
        self.num_bins = num_bins
        self.exp_base = num_bins ** (1 / num_bins) if exp_base == -1 else exp_base
        self.invert = invert

        if invert:
            self.exp_base = 1 / self.exp_base

        self.range_size = self.range_end - self.range_start
        self.bin_size = (self.range_size * (1 - self.exp_base)) / (1 - self.exp_base ** self.num_bins)
        self.breaks = list(map(lambda x: self.range_start + (self.bin_size * ((1-(self.exp_base ** x)) / (1-self.exp_base))),
                               list(range(0, self.num_bins + 1))))

    def getbin(self, value):
        """Returns the bin a number falls into."""
        assert (self.range_start <= value and value <= self.range_end)

        return math.floor(math.log(1 - (value - self.range_start) * (1-self.exp_base)/self.bin_size,
                                   self.exp_base))

