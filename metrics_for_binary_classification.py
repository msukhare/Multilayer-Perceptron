class metrics_for_binary_classification:

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    ## Y == class, pred == prediction make by classification, them shape == (nb_exemple, nb_class)
    def confused_matrix_soft_max(self, pred, Y, put_metrics):
        for i in range(int(Y.shape[0])):
            if (Y[i][0] == 1 and pred[i][0] > pred[i][1]):
                self.tp += 1
            elif (Y[i][0] == 0 and pred[i][0] < pred[i][1]):
                self.tn += 1
            elif (Y[i][0] == 1 and pred[i][0] < pred[i][1]):
                self.fn += 1
            elif (Y[i][0] == 0 and pred[i][0] > pred[i][1]):
                self.fp += 1
        if (put_metrics == 1):
            print("tp = ", self.tp, ", tn = ", self.tn, ", fp = ", self.fp,\
                    ", fn = ", self.fn)
            print("accuracy = ", self.get_accuracy(Y.shape[0]))
            print("precision = ", self.get_precision())
            print("recall = ", self.get_recall())
            print("f1 = ", self.get_f1())
            print("classification error = ", \
                    self.get_classification_error(Y.shape[0]))
            print("false alarm rate = ", self.get_false_alarm_rate())
            print("miss rate = ", self.get_miss_rate())
            print("kappa_cohen = ", self.kappa_cohen())

    def get_accuracy(self, nb_exemple):
        if (nb_exemple == 0):
            print("need more exemple for accuracy ")
            return (0)
        return (((self.tp + self.tn) / nb_exemple))

    def get_precision(self):
        if ((self.tp + self.fp) == 0):
            print("need more tp or fp for precision ")
            return (0)
        return ((self.tp / (self.tp + self.fp)))

    def get_recall(self):
        if ((self.tp + self.fn) == 0):
            print("need more tp or fn for recall ")
            return (0)
        return ((self.tp / (self.tp + self.fn)))

    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()
        if ((precision + recall) == 0):
            print("need more precision or recall for f1 ")
            return (0)
        return (((2 * (precision * recall)) / (precision + recall)))

    def get_classification_error(self, nb_exemple):
        if (nb_exemple == 0):
            print("need more nb_exemple for class_error ")
            return (0)
        return (((self.fp + self.fn) / nb_exemple))

    def get_false_alarm_rate(self):
        if ((self.fp + self.tn) == 0):
            print("need more fp or tn for false alarm rate ")
            return (0)
        return ((self.fp / (self.fp + self.tn)))

    def get_miss_rate(self):
        if ((self.tp + self.fn) == 0):
            print("need more tp or fn for miss rate")
            return (0)
        return ((self.fn / (self.tp + self.fn)))

    def kappa_cohen(self):
        total = (self.tn + self.tp + self.fp + self.fn)
        if (total == 0):
            print("need more tn or tp or fp or fn for kappa_cohen")
            return (0)
        paccord = ((self.tp + self.tn) / total)
        pyes = (((self.tp + self.fn) / total) * ((self.tp + self.fp) / total))
        pno = (((self.fp + self.tn) / total) * ((self.fn + self.tn) / total))
        phasard = pyes + pno
        return (((paccord - phasard) / (1 - phasard)))
