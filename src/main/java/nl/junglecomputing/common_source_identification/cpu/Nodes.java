/*
 * Copyright 2018 Vrije Universiteit Amsterdam, The Netherlands
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package nl.junglecomputing.common_source_identification.cpu;

import java.util.Hashtable;

/* Nodes is a doubly linked list for implementing LRU for a cache.  For fast access of nodes based on the index, we also contain
 * a mapping from index to node in the linked list.
 */
class Nodes {

    private Hashtable<Integer, Node> nodes;
    private Node head;
    private Node tail;

    // create an empty linked list
    Nodes() {
        this.head = null;
        this.tail = null;
        this.nodes = new Hashtable<Integer, Node>();
    }

    /* 
     * Add index index to the head.  If the node is not in the list, create a new one.
     */
    void addToHead(int index) {
        Node node = nodes.get(index);
        if (node == null) {
            addNewNodeToHead(index);
        } else {
            moveExistingNodeToHead(node);
        }
        // registered in map and linked list
    }

    // Add a specific node to the head
    void addToHead(Node node) {
        moveToHead(node);
        nodes.put(node.index, node);
        // registered in map and linked list
    }

    // get the index of the tail
    int getTailIndex() {
        return tail.index;
    }

    // remove an element from the linked list if it is there.
    Node remove(int index) {
        Node node = nodes.get(index);
        if (node == null) {
            return null;
        }
        remove(node);
        nodes.remove(index);

        return node;
        // registered in map and linked list
    }

    // return the number of items in the linked list
    int size() {
        return nodes.size();
    }

    // private methods

    private void addNewNodeToHead(int index) {
        Node node = new Node(index);
        nodes.put(index, node);
        moveToHead(node);
    }

    private void moveToHead(Node node) {
        // assumes that node.previous and node.next can be overwritten
        // assumes that no other nodes are pointing to this node
        node.next = head;
        node.previous = null;
        if (head == null) {
            tail = node;
        } else {
            head.previous = node;
        }
        head = node;
    }

    private void moveExistingNodeToHead(Node node) {
        remove(node);
        moveToHead(node);
    }

    private void remove(Node node) {
        if (node == head) {
            removeHead();
        } else if (node == tail) {
            tail = node.previous;
            tail.next = null;
        } else {
            node.previous.next = node.next;
            node.next.previous = node.previous;
        }
        node.next = null;
        node.previous = null;
    }

    private void removeHead() {
        if (head.next == null) {
            head = null;
            tail = null;
        } else {
            head.next.previous = null;
            head = head.next;
        }
    }

    @Override
    public String toString() {
        Node next = head;
        StringBuilder sb = new StringBuilder("[");
        while (next != null) {
            sb.append(next);
            if (next.next != null) {
                sb.append(", ");
            }
            next = next.next;
        }
        sb.append("]\n");
        sb.append(nodes);
        return sb.toString();
    }
}
