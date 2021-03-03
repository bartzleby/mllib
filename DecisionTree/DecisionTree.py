#
#
#
#
#

class DecisionTreeNode(object):
    "Decision tree node."

    def __init__(self, attribute=None, values=None, label=None):
        self.attribute = attribute
        self.values = values
        self.label = label

        self.branches = {}


    def __repr__(self):
        '''Currently a little wonky:
        returns an attribute if a splitting node
        returns a label if not
        '''
        if self.attribute is None:
          return self.label
        return self.attribute

    def add_branch(self, node, key):
        assert isinstance(node, DecisionTreeNode)
        self.branches.update({key : node})

    def classify(self, value):
        '''Return the next node
        If value is not in branches,
        we have yet to return some error.
        '''
        return self.branches[value]


def classify(dtroot, example, attributes):
  '''Calissify input example from dtroot.
  Currently does not handle errors, e.g. 
  inputs not corresponding properly.

  params:
    dtroot -- root nod of decision tree ()
    example -- list of attribute values
          corrsponding to attribute input
    attributes -- list of attributes
  '''
  # so we don't accidentally destroy input list
  attrs = list(attributes)
  #print("receiving: ", example)
  while dtroot.label is None:
    test_attr = dtroot.attribute
    tai = attrs.index(test_attr)
    dtroot = dtroot.classify(example[tai])
    del attrs[tai]
    del example[tai]

  return dtroot.label

