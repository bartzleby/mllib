#
#
#
#
#

class DecisionTreeNode(object):
    "Decision tree node."

    def __init__(self, attribute=None, values = None, label=None):
        self.attribute = attribute
        self.values = values
        self.label = label

        self.branches = {}

        
    def __repr__(self):
        if self.attribute is None:
          return "leaf"
        return self.attribute

    def add_branch(self, node, key):
        assert isinstance(node, DecisionTreeNode)
        self.branches.update({key : node})
