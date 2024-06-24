class Piece:
    def __init__(self, name, rank, color, valid_positions):
        self.name = name
        self.rank = rank
        self.color = color
        self.position = None
        self.valid_positions = valid_positions

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    def get_name(self):
        return self.name

    def get_rank(self):
        return self.rank

    def get_color(self):
        return self.color

    def get_valid_positions(self):
        return self.valid_positions
