import numpy as np


def get_event_list(y):
    event_list = []
    start_time = 0
    end_time = 0
    while end_time < len(y):
        start_time = end_time
        label = y[start_time]
        end_time = get_event_end(y, start_time, label)
        event_list.append((start_time, end_time, label))
    return event_list


def get_event_end(y, time, label):
    while time < len(y) and y[time] == label:
        time += 1
    return time


def intersection_over_union(event1, event2, verbose=False):
    intersection = min(event1[1], event2[1]) - max(event1[0], event2[0])
    union = max(event1[1], event2[1]) - min(event1[0], event2[0])
    if verbose:
        print(
            f"Events {event1}, {event2} have intersection {intersection} and union {union} and IoO {intersection / union}")
    return intersection / union


def count_events(y):
    curr_label = y[0]
    count = 1
    for i in range(len(y)):
        if curr_label != y[i]:
            count += 1
            curr_label = y[i]
    return count


class EventAccuracy:

    def __init__(self, y_true, y_pred, threshold=[0.5,0.3,0.3], nb_classes=3, verbose=False):
        self.accuracy = 0
        self.verbose = verbose
        self.thresholds = threshold
        self.y_true = y_true
        self.y_pred = y_pred
        self.nb_true_events = count_events(self.y_true)
        self.nb_pred_events = count_events(self.y_pred)
        self.true_events = get_event_list(self.y_true)
        self.pred_events = get_event_list(self.y_pred)
        self.T = 0
        self.F = 0
        if self.verbose:
            print("GT:", np.array(self.y_true))
            #print("GT:", self.true_events)
            #print(f"GT has {self.nb_true_events} events")
            print(f"PR:", np.array(self.y_pred))
            #print("PR:", self.pred_events)
            #print(f"PR has {self.nb_pred_events} events")

    def eval(self):
        # go over all events end check whether we hit the event, naive, just check at center of event
        # TODO: come up with more advanced search of whether event was found or not, fixation should overlap, sacc and blk not as much
        true_iter = iter(self.true_events)
        true_event = next(true_iter, None)
        pred_index = 0
        while true_event is not None:
            label = true_event[2]
            event_len = true_event[1] - true_event[0]
            center = int((true_event[0] + true_event[1]) / 2)
            while pred_index < len(self.pred_events) and not self.pred_events[pred_index][0] <= center < self.pred_events[pred_index][1]:
                pred_index += 1
            if self.pred_events[pred_index][2] == true_event[2]:
                int_over_union = intersection_over_union(true_event, self.pred_events[pred_index])
                if int_over_union >= self.thresholds[label]:
                    self.T += 1
                    #print(f"HIT: {true_event, self.pred_events[pred_index]} at index {center}")
                else:
                    #print(f"MIS: {true_event, self.pred_events[pred_index]} at index {center}")
                    self.F += 1
            else:
                #print(f"MIS: {true_event, self.pred_events[pred_index]} at index {center}")
                pass
            true_event = next(true_iter, None)

        if self.T + self.F > 0:
            self.accuracy = self.T / self.nb_true_events

        if self.verbose:
            print(f"Event Accuracy: {self.accuracy}")
        return self.accuracy 


class EventMetric:

    def __init__(self, y_true, y_pred, threshold=.5, nb_classes=3):
        self.y_true = y_true
        self.y_pred = y_true
        self.length = len(y_true)
        self.curr_event_start = 0
        self.curr_event_end
        self.curr_event_label = -1
        self.threshold = threshold
        self.true_events = get_event_list(self.y_true)
        self.pred_events = get_event_list(self.y_pred)
        self.TP = [0 for i in range(nb_classes)]
        self.TN = [0 for i in range(nb_classes)]
        self.FP = [0 for i in range(nb_classes)]
        self.FN = [0 for i in range(nb_classes)]

    def eval(self):
        for i, event in enumerate(self.true_events):
            pass

    def intersection_over_union(pred_events):
        min_time = min(self.curr_event_start, pred_events[0][0])  # left window for union
        max_time = max(self.curr_event_end, pred_events[-1][1])  # right window for union
        intersection = sum([min(end, ende) - max(start, begin) for (start, end, label) in pred_events])
        union = max(self.curr_event_end, pred_events[-1][1]) - min(self.curr_event_start, pred_events[0][0])
        return intersection / union

    def next_event_time(self):
        self.curr_event_start = self.curr_event_end
        while self.y_true[self.curr_event_end] == self.curr_event_label:
            self.curr_event_end += 1
        self.curr_event_label = self.y_true[self.curr_event_start]


#y_sim = np.concatenate([np.repeat(np.random.randint(0,3), 5) for i in range(100)])
y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1])
#y_true = np.concatenate([np.repeat(np.random.randint(0,3), 5) for i in range(100)])
y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0])
y_dummy = np.array([1 for i in range(len(y_true))])
# metric = EventMetric(y_true=y_true, y_pred=y_dummy)
metric = EventAccuracy(y_true=y_true, y_pred=y_pred)
metric.eval()
