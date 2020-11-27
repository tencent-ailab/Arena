import math
import random
import sys

import numpy as np
import pygame

from .pong2p_utils import Rect


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class PongGame():
    def __init__(self,
                 has_double_players=False,
                 window_size=(160, 160),
                 top_border_thickness=34,
                 ball_width=2,
                 ball_height=4,
                 ball_speed=1,
                 bat_offset=16,
                 bat_width=4,
                 bat_height=16,
                 bat_speed=1,
                 max_num_rounds=20,
                 max_step_per_round=1000):

        self._max_num_rounds = max_num_rounds
        self._has_double_players = has_double_players
        self._max_step_per_round = max_step_per_round
        self._num_rounds = 0
        self._bat_offset = bat_offset
        self._random_flag = False

        self._arena = Arena(window_size, top_border_thickness)
        self._ball = Ball(ball_width, ball_height, ball_speed)
        self._left_bat = Bat(bat_width, bat_height, bat_speed)
        if self._has_double_players:
            self._right_bat = Bat(bat_width, bat_height, bat_speed)
        else:
            self._right_bat = AutoBat(bat_width, bat_height, bat_speed)

        self._scoreboard = Scoreboard(20, 8, font_size=20)

        self.reset_game()

    def set_seed(self, seed):
        if seed is not None:
            self._random_flag = True
            random.seed(seed)

    def step(self, left_bat_action, right_bat_action):
        self._num_steps += 1
        self._left_bat.move(self._arena, left_bat_action)
        if self._has_double_players:
            self._right_bat.move(self._arena, right_bat_action)
        else:
            self._right_bat.move(self._arena, self._ball)
        self._ball.move(self._arena, self._left_bat, self._right_bat)

        if self._ball.left_out_of_arena(self._arena):
            self._score_right += 1
            rewards = (-1, 1)
            self._reset_round(-1)
        elif self._ball.right_out_of_arena(self._arena):
            self._score_left += 1
            rewards = (1, -1)
            self._reset_round(1)
        elif self._num_steps > self._max_step_per_round:
            print("Time out to be a tie round.")
            rewards = (0, 0)
            self._reset_round(0)
        else:
            rewards = (0, 0)

        #if (self._score_left >= self._max_num_rounds or
        #        self._score_right >= self._max_num_rounds):
        if self._num_rounds >= self._max_num_rounds:
            done = True
        else:
            done = False
        return rewards, done

    def _reset_round(self, start_dir=0):
        self._reset_ball(start_dir)
        self._num_rounds += 1
        self._num_steps = 0

    def _reset_ball(self, start_dir):
        if self._random_flag:
            centerx = self._arena.centerx - self._ball.width() / 2.0
            #centerx = random.uniform(self._arena.left, 
            #        self._arena.right - self._ball.width())
            centery = random.uniform(self._arena.top,
                    self._arena.bottom - self._ball.height())
            speed = float(self._ball.speed())
            speedx = -speed
            rand = random.randint(0, 2)
            speedy = speed * rand
            speedx *= (1 if random.random() < 0.9 else 3)
        else:
            centerx = self._arena.centerx - self._ball.width() / 2.0
            centery = self._arena.centery - self._ball.height() / 2.0
            speed = float(self._ball.speed())
            speedy = speedx = speed

        if start_dir == 0:
            speedx = random.choice([speedx, -speedx])
        elif start_dir < 0:
            speedx *= -1.0

        speedy = random.choice([speedy, -speedy])
        self._ball.reset(centerx, centery, speedx, speedy)

    def _reset_bat(self):
        if self._random_flag:
            lefty = random.uniform(self._arena.top - self._left_bat.height(),
                    self._arena.bottom)
            righty = random.uniform(self._arena.top - self._right_bat.height(),
                    self._arena.bottom)
            #rand = random.uniform(0.5, 1.0)
            #rand = 1.0
            lefts = self._ball.speed() * random.uniform(1.0, 5.0)
            rights = self._ball.speed() * random.uniform(1.0, 5.0)
        else:
            lefty = self._arena.centery - self._left_bat.height() / 2.0
            righty = self._arena.centery - self._right_bat.height() / 2.0
            lefts = self._left_bat.max_speed()
            rights = self._right_bat.max_speed()

        leftx = self._bat_offset
        rightx = self._arena.right - self._right_bat.width() - self._bat_offset

        self._left_bat.reset(leftx, lefty, lefts)
        self._right_bat.reset(rightx, righty, rights)

    def reset_game(self):
        self._score_left, self._score_right = 0, 0
        self._reset_round()
        self._reset_bat()
        self._num_rounds = 0

    def draw(self, surface):
        self._arena.draw(surface)
        self._ball.draw(surface)
        self._left_bat.draw(surface)
        self._right_bat.draw(surface)

    def draw_scoreboard(self, surface):
        self._scoreboard.draw(surface, self._score_left, self._score_right)


class Arena(pygame.sprite.Sprite):
    def __init__(self, window_size, top_border_thickness):
        window_width, window_height = window_size
        self._rect = pygame.Rect(0, top_border_thickness, window_width,
                                 window_height)

    def draw(self, surface):
        surface.fill(WHITE)
        pygame.draw.rect(surface, BLACK, self._rect)

    @property
    def left(self):
        return self._rect.left

    @property
    def right(self):
        return self._rect.right

    @property
    def top(self):
        return self._rect.top

    @property
    def bottom(self):
        return self._rect.bottom

    @property
    def centerx(self):
        return self._rect.centerx

    @property
    def centery(self):
        return self._rect.centery


class Bat():
    def __init__(self, width, height, speed):
        self._rect = Rect(0, 0, width, height)
        self._max_speed = speed
        self._speed = 0.0

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self._draw_rect)

    def move(self, arena, action):
        if action == 0:
            delta_v = 0.1 * self._max_speed
            if abs(delta_v) > abs(self._speed):
                self._speed = 0.0
            else:
                self._speed -= delta_v * np.sign(self._speed)
        elif action == 1:
            delta_v = 0.5 * self._max_speed
            if abs(delta_v) > abs(self._speed):
                self._speed = 0.0
            else:
                self._speed -= delta_v * np.sign(self._speed)
        elif action == 2:
            self._speed += 0.25 * self._max_speed
            self._speed = min(self._speed, self._max_speed)
        elif action == 3:
            self._speed -= 0.25 * self._max_speed
            self._speed = max(self._speed, -self._max_speed)
        elif action == 4:
            self._speed += 0.1 * self._max_speed
            self._speed = min(self._speed, self._max_speed)
        elif action == 5:
            self._speed -= 0.1 * self._max_speed
            self._speed = max(self._speed, -self._max_speed)

        self._rect.add_y(self._speed)
        if self._rect.bottom() > arena.bottom:
            self._rect.add_y(arena.bottom - self._rect.bottom())
            self._speed = 0.0
        elif self._rect.top() < arena.top:
            self._rect.add_y(arena.top - self._rect.top())
            self._speed = 0.0

        self._draw_rect = self._rect.rect()

    def reset(self, x, y, speed):
        self._rect.set_pos(x, y)
        self._max_speed = speed

        self._draw_rect = self._rect.rect()

    def left(self):
        return self._rect.left()

    def right(self):
        return self._rect.right()

    def top(self):
        return self._rect.top()

    def bottom(self):
        return self._rect.bottom()

    def width(self):
        return self._rect.width()

    def height(self):
        return self._rect.height()

    def centerx(self):
        return self._rect.centerx()

    def centery(self):
        return self._rect.centery()

    def speed(self):
        return self._speed

    def max_speed(self):
        return self._max_speed


class AutoBat(Bat):
    def move(self, arena, ball):
        self._speed = ball.speed() * 1.25
        #If ball is moving away from paddle, center bat
        if ball.speed_x() < 0:
            if self._rect.centery() < arena.centery:
                self._rect.add_y(self._speed)
            elif self._rect.centery() > arena.centery:
                self._rect.add_y(-self._speed)
        #if ball moving towards bat, track its movement.
        elif ball.speed_x() > 0:
            if self._rect.centery() < ball.centery():
                self._rect.add_y(self._speed)
            else:
                self._rect.add_y(-self._speed)

        if self._rect.bottom() > arena.bottom:
            self._rect.add_y(arena.bottom - self._rect.bottom())
        elif self._rect.top() < arena.top:
            self._rect.add_y(arena.top - self._rect.top())

        self._draw_rect = self._rect.rect()


class Scoreboard():
    def __init__(self, x, y, font_size=20):
        self._x = x
        self._y = y
        self._font = pygame.font.Font('freesansbold.ttf', font_size)

    def draw(self, surface, score_left, score_right):
        result_surf = self._font.render('%d    :    %d' %
                                        (score_left, score_right), True, BLACK)
        w = result_surf.get_rect().width
        h = result_surf.get_rect().height
        sw = surface.get_width()
        rect = result_surf.get_rect()
        rect.topleft = ((sw - w) / 2.0, self._y)
        surface.blit(result_surf, rect)


class Ball():
    def __init__(self, w, h, speed):
        self._rect = Rect(0, 0, w, h)
        self._speed = speed

    def reset(self, x, y, speed_x, speed_y):
        self._rect.set_pos(x, y)
        self._speed_x = speed_x
        self._speed_y = speed_y
        self._draw_rect = self._rect.rect()

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self._draw_rect)

    def move(self, arena, left_bat, right_bat):
        px = self._rect.x()
        py = self._rect.y()

        self._rect.add_pos(self._speed_x, self._speed_y)

        # Arena
        if self._speed_y < 0 and self._rect.top() <= arena.top:
            self._reflect()
            self._rect.set_top(arena.top +
                    arena.top - self._rect.top())
            py = arena.top * 2 - py
        elif self._speed_y > 0 and self._rect.bottom() >= arena.bottom:
            self._reflect()
            self._rect.set_bottom(arena.bottom +
                    arena.bottom - self._rect.bottom())
            py = (arena.bottom - self._rect.height()) * 2 - py

        # Bat
        class Point():
            def __init__(self, x, y):
                self.x = x
                self.y = y

        def toleft(p, q, r):
            return (q.x * r.y + p.x * q.y +
                    p.y * r.x - p.x * r.y -
                    q.y * r.x - p.y * q.x)

        def intersect(p1, p2, p3, p4):
            if (toleft(p1, p2, p3) * toleft(p1, p2, p4) <= 0 and
                toleft(p3, p4, p1) * toleft(p3, p4, p2) <= 0):
                return True
            return False

        def collision(rect, p1, p2):
            w = rect.width()
            h = rect.height()
            rx = rect.x()
            ry = rect.y()
            track_list = [
                    (Point(px, py), Point(rx, ry)),
                    (Point(px + w, py), Point(rx + w, ry)),
                    (Point(px, py + h), Point(rx, ry + h)),
                    (Point(px + w, py + h), Point(rx + w, ry + h))]
            for p3, p4 in track_list:
                if intersect(p1, p2, p3, p4):
                    return True
            return False

        def fast_test(r1, r2):
            xmin = max(r1.left(), r2.left())
            ymin = max(r1.top(), r2.top())
            xmax = min(r1.right(), r2.right())
            ymax = min(r1.bottom(), r2.bottom())
            if (xmin <= xmax) and (ymin <= ymax):
                return True
            return False

        if self._speed_x < 0:
            fast_rect = Rect(min(self._rect.x(), px),
                    min(self._rect.y(), py),
                    abs(self._speed_x) + self.width(),
                    abs(self._speed_y) + self.height())
            if fast_test(fast_rect, left_bat._rect):
                collide = False
                onbat_y = 0.0
                bounce_range = (left_bat.height() + self.height()) / 2.0
                if collision(self._rect,
                        Point(left_bat.right(), left_bat.top()),
                        Point(left_bat.right(), left_bat.bottom())):
                    onbat_y = (px - left_bat.right()) / self._speed_x * self._speed_y + \
                            py + self.height() / 2.0 - left_bat.centery()
                    collide = True
                elif self._speed_y > 0 and collision(self._rect,
                        Point(left_bat.left(), left_bat.top()),
                        Point(left_bat.right(), left_bat.top())):
                    onbat_y = -bounce_range
                    collide = True
                elif self._speed_y < 0 and collision(self._rect,
                        Point(left_bat.left(), left_bat.bottom()),
                        Point(left_bat.right(), left_bat.bottom())):
                    onbat_y = bounce_range
                    collide = True

                if collide:
                    self._rect.set_left(left_bat.right())
                    self._rect.set_centery(onbat_y + left_bat.centery())
                    angle = onbat_y / (left_bat.height() / 2.0)

                    still = True if left_bat.speed() == 0.0 else False
                    self._bounce('left', angle, still)

        elif self._speed_x > 0:
            fast_rect = Rect(min(self._rect.x(), px),
                    min(self._rect.y(), py),
                    abs(self._speed_x) + self.width(),
                    abs(self._speed_y) + self.height())
            if fast_test(fast_rect, right_bat._rect):
                collide = False
                onbat_y = 0.0
                bounce_range = (right_bat.height() + self.height()) / 2.0
                if collision(self._rect,
                        Point(right_bat.left(), right_bat.top()),
                        Point(right_bat.left(), right_bat.bottom())):
                    onbat_y = (px - right_bat.left()) / self._speed_x * self._speed_y + \
                            py + self.height() / 2.0 - right_bat.centery()
                    collide = True
                elif self._speed_y > 0 and collision(self._rect,
                        Point(right_bat.left(), right_bat.top()),
                        Point(right_bat.right(), right_bat.top())):
                    onbat_y = -bounce_range
                    collide = True
                elif self._speed_y < 0 and collision(self._rect,
                        Point(right_bat.left(), right_bat.bottom()),
                        Point(right_bat.right(), right_bat.bottom())):
                    onbat_y = bounce_range
                    collide = True

                if collide:
                    self._rect.set_right(right_bat.left())
                    self._rect.set_centery(onbat_y + right_bat.centery())
                    angle = onbat_y / (right_bat.height() / 2.0)
                    still = True if left_bat.speed() == 0.0 else False
                    self._bounce('right', angle, still)

        self._draw_rect = self._rect.rect()


    def left_out_of_arena(self, arena):
        return True if self._rect.left() < arena.left else False

    def right_out_of_arena(self, arena):
        return True if self._rect.right() > arena.right else False

    def _reflect(self):
        self._speed_y *= -1.0

    def _bounce(self, side, angle, still):
        if not still and (angle >= 1.0 or angle <= -1.0):
            self._speed_x = 3.0 * self._speed
        else:
            self._speed_x = self._speed

        if angle > 0.25:
            self._speed_y = self._speed * (1.0 if still else 2.0)
        elif angle > -0.25:
            self._speed_y = 0.0
        else:
            self._speed_y = -self._speed * (1.0 if still else 2.0)

        if side == 'right':
            self._speed_x *= -1.0

    def width(self):
        return self._rect.width()

    def height(self):
        return self._rect.height()

    def speed(self):
        return self._speed

    def speed_x(self):
        return self._speed_x

    def speed_y(self):
        return self._speed_y

    def centerx(self):
        return self._rect.centerx()

    def centery(self):
        return self._rect.centery()


#Main function
def main():
    pygame.init()
    pygame.display.set_caption('Pong')
    pygame.mouse.set_visible(0)  # make cursor invisible
    surface = pygame.display.set_mode((160, 210))
    fps_clock = pygame.time.Clock()

    game = PongGame()

    while True:  #main game loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # mouse movement commands
            #elif event.type == MOUSEMOTION:
            #    game.bats['user'].move(event.pos)

        action = random.randint(0, 5)
        print(action)
        _, done = game.step(action, 0)
        if done:
            game.reset_game()
        game.draw(surface)
        pygame.display.update()
        fps_clock.tick(120)


if __name__ == '__main__':
    main()
