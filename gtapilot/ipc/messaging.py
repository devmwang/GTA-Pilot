from collections import deque
import threading

class PubMaster:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, topic, subscriber):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(subscriber)

    def publish(self, topic, message):
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                subscriber(message)

class SubMaster:
    def __init__(self):
        self.topics = {}

    def add_topic(self, topic):
        self.topics[topic] = None

    def receive(self, topic, message):
        if topic in self.topics:
            self.topics[topic] = message

class Message:
    def __init__(self, topic, data):
        self.topic = topic
        self.data = data

class MessagingSystem:
    def __init__(self):
        self.pub_master = PubMaster()
        self.sub_master = SubMaster()

    def publish(self, topic, data):
        message = Message(topic, data)
        self.pub_master.publish(topic, message)

    def subscribe(self, topic, callback):
        self.pub_master.subscribe(topic, callback)
        self.sub_master.add_topic(topic)