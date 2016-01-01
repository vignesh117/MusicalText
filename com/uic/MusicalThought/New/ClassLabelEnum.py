__author__ = 'vignesh'

class ClassLabelEnum(object):

    m1 = 0 # Corresponds to /
    m2 = 1 # Corresponds to //
    space = 2 # Corresponds to NM
    label = None
    def __init__(self, c):

        if c == '/':
            self.label = self.m1
        elif c =='//':
            self.label = self.m2
        elif c == 'NM':
            self.label = self.space
        else:
            self.label = c

    def get_label(self):
        return self.label


