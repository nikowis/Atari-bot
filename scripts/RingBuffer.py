import random


class RingBuf:
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.size = size
        self.end = 0
        self.total = 0

    def append(self, element):
        self.total = self.total + 1
        if self.total > self.size:
            self.total = self.size
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def get_batch(self, batch_size):
        indices = random.sample(range(self.total), batch_size)
        mylist = []
        for index in indices:
            mylist.append(self.data[index])
        return mylist

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
