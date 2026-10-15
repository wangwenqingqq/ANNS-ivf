"""Small sequential predecessor controls; no concurrent-node lifetime protocol."""
from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass


@dataclass
class Node:
    key: int
    left: Node | None = None
    right: Node | None = None
    height: int = 1


class Index:
    def __init__(self, kind):
        assert kind in ("chain", "avl", "array")
        self.kind, self.root, self.array = kind, None, []
        self.present = set()
        self.rotations = 0

    @staticmethod
    def height(node):
        return node.height if node else 0

    def refresh(self, node):
        node.height = 1 + max(self.height(node.left), self.height(node.right))
        return node

    def rotate(self, node, left):
        self.rotations += 1
        child = node.right if left else node.left
        if left:
            node.right, child.left = child.left, node
        else:
            node.left, child.right = child.right, node
        self.refresh(node)
        return self.refresh(child)

    def balance(self, node):
        self.refresh(node)
        delta = self.height(node.left) - self.height(node.right)
        if delta > 1:
            if self.height(node.left.left) < self.height(node.left.right):
                node.left = self.rotate(node.left, True)
            return self.rotate(node, False)
        if delta < -1:
            if self.height(node.right.right) < self.height(node.right.left):
                node.right = self.rotate(node.right, False)
            return self.rotate(node, True)
        return node

    def insert_tree(self, node, key):
        if node is None:
            return Node(key)
        if key < node.key:
            node.left = self.insert_tree(node.left, key)
        else:
            node.right = self.insert_tree(node.right, key)
        return self.balance(node)

    def delete_tree(self, node, key):
        if key < node.key:
            node.left = self.delete_tree(node.left, key)
        elif key > node.key:
            node.right = self.delete_tree(node.right, key)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = node.right
            while successor.left:
                successor = successor.left
            node.key = successor.key
            node.right = self.delete_tree(node.right, successor.key)
        return self.balance(node)

    def add(self, key):
        assert key not in self.present
        self.present.add(key)
        if self.kind == "avl":
            self.root = self.insert_tree(self.root, key)
        elif self.kind == "array":
            # Pressure/build workloads append; mixed correctness tests also insert.
            if not self.array or key > self.array[-1]:
                self.array.append(key)
            else:
                self.array.insert(bisect_left(self.array, key), key)
        elif self.root is None or key > self.root.key:
            self.root = Node(key, self.root)
        else:
            previous = self.root
            while previous.left and previous.left.key > key:
                previous = previous.left
            previous.left = Node(key, previous.left)

    def remove(self, key):
        assert key in self.present
        self.present.remove(key)
        if self.kind == "avl":
            self.root = self.delete_tree(self.root, key)
        elif self.kind == "array":
            self.array.pop(bisect_left(self.array, key))
        elif self.root.key == key:
            self.root = self.root.left
        else:
            previous = self.root
            while previous.left.key != key:
                previous = previous.left
            previous.left = previous.left.left

    def predecessor(self, timestamp):
        visits, answer = 0, None
        if self.kind == "array":
            low, high = 0, len(self.array)
            while low < high:
                mid = (low + high) // 2
                visits += 1
                if self.array[mid] <= timestamp:
                    answer, low = self.array[mid], mid + 1
                else:
                    high = mid
            return answer, visits
        node = self.root
        while node:
            visits += 1
            if self.kind == "chain":
                if node.key <= timestamp:
                    return node.key, visits
                node = node.left
            elif node.key <= timestamp:
                answer, node = node.key, node.right
            else:
                node = node.left
        return answer, visits

    def check(self):
        if self.kind == "array":
            assert self.array == sorted(self.present)
            return
        seen = set()
        if self.kind == "chain":
            node, previous = self.root, float("inf")
            while node:
                assert node.key < previous and node.key not in seen
                seen.add(node.key)
                previous, node = node.key, node.left
        else:
            def visit(node, low, high):
                if node is None:
                    return 0
                assert low < node.key < high and node.key not in seen
                seen.add(node.key)
                left = visit(node.left, low, node.key)
                right = visit(node.right, node.key, high)
                assert abs(left - right) <= 1
                assert node.height == 1 + max(left, right)
                return node.height
            visit(self.root, float("-inf"), float("inf"))
        assert seen == self.present
