
class AgentAlways:
    """
    Always repair agent
    """
    def __init__(self):
        pass

    def take_action(self, state):
        return 1

class AgentNever:
    """
    Never repair agent
    """
    def __init__(self):
        pass

    def take_action(self, state):
        return 0
