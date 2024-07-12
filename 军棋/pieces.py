class Piece:
    def __init__(self, name, rank, color, valid_positions):
        self.name = name
        self.rank = rank
        self.color = color
        self.position = None
        self.valid_positions = valid_positions
        self.moved = False  # 新增的属性，记录棋子是否移动
        self.revealed = False

    def set_position(self, position):
        self.position = position
        self.moved = True  # 设置位置时更新移动状态

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
    
    def is_revealed(self):
        return self.revealed
    
    def set_revealed(self, revealed):
        self.revealed = revealed

    def has_moved(self):
        return self.moved  # 返回棋子是否移动的状态