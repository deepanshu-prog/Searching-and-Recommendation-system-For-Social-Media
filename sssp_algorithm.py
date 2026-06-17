import math
from collections import namedtuple
from enum import Enum

FLOAT_TOLERANCE = 1e-9


class MessageType(Enum):
    DISTANCE_UPDATE = "DistanceUpdate"
    SET_TO_INFINITY = "SetToInfinity"
    DISTANCE_QUERY = "DistanceQuery"
    ADD_TO_SUCCESSOR = "AddToSuccessor"
    REMOVE_FROM_SUCCESSOR = "RemoveFromSuccessor"


Message = namedtuple('Message', ['from_id', 'to_id', 'type', 'body'])


class User:
    def __init__(self, user_id, username, location):
        self.id = user_id
        self.username = username
        self.location = location
        self.distance = math.inf
        self.predecessor = None
        self.successors = set()
        self.marked_as_infinity = False
        self.out_edges = {}
        self.in_edges = {}

    def reset_sssp(self):
        self.distance = math.inf
        self.predecessor = None
        self.successors = set()
        self.marked_as_infinity = False

    def on_message(self, msg, message_queue, all_users):
        if msg.type == MessageType.DISTANCE_UPDATE:
            new_distance = msg.body['distance']
            if new_distance < (self.distance - FLOAT_TOLERANCE):
                if self.predecessor and self.predecessor in all_users:
                    message_queue.append(Message(self.id, self.predecessor, MessageType.REMOVE_FROM_SUCCESSOR, {}))

                self.distance = new_distance
                self.predecessor = msg.from_id

                if self.predecessor in all_users:
                    message_queue.append(Message(self.id, self.predecessor, MessageType.ADD_TO_SUCCESSOR, {}))

                for friend_id, weight in self.out_edges.items():
                    if friend_id != self.predecessor and friend_id in all_users:
                        message_queue.append(Message(self.id, friend_id, MessageType.DISTANCE_UPDATE, {'distance': self.distance + weight}))

        elif msg.type == MessageType.SET_TO_INFINITY:
            if self.marked_as_infinity or self.distance == math.inf:
                return
            self.distance = math.inf
            self.predecessor = None
            self.marked_as_infinity = True
            successors_to_notify = list(self.successors)
            self.successors.clear()
            for successor_id in successors_to_notify:
                if successor_id in all_users:
                    successor_node = all_users.get(successor_id)
                    if successor_node and successor_node.predecessor == self.id:
                        message_queue.append(Message(self.id, successor_id, MessageType.SET_TO_INFINITY, {}))

        elif msg.type == MessageType.DISTANCE_QUERY:
            if self.distance != math.inf:
                weight_to_querier = self.out_edges.get(msg.from_id)
                if weight_to_querier is not None and msg.from_id in all_users:
                    message_queue.append(Message(self.id, msg.from_id, MessageType.DISTANCE_UPDATE, {'distance': self.distance + weight_to_querier}))

        elif msg.type == MessageType.ADD_TO_SUCCESSOR:
            if msg.from_id in all_users:
                self.successors.add(msg.from_id)

        elif msg.type == MessageType.REMOVE_FROM_SUCCESSOR:
            self.successors.discard(msg.from_id)
