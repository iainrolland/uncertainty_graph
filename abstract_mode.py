class Data:
    def __init__(self):
        self.array = None

    def normalise(self, nodata):
        """Sets pixels with nodata value to zero then normalises each channel to between 0 and 1"""
        self.array[self.array == nodata] = 0
        self.array = (self.array - self.array.min(axis=(1, 2))[:, None, None]) / (
            (self.array.max(axis=(1, 2)) - self.array.min(axis=(1, 2)))[:, None, None])
