import turtle
import random
import math
import sys

# if RNG seed is None, defaults to system time
rng_seed = None
number_iterations = 50
number_cities = 30


def main():
    # Just to be sure we don't accidentally
    # create a new rng_seed if we edit code later
    global rng_seed

    get_arguments()

    # Could add a fixed seed for debugging
    random.seed(rng_seed)
    citylocations = []

    # Generating the Cities
    for i in range(0, number_cities):
        citylocations.append(
            (random.randint(-299, 299),
             random.randint(-299, 299)))

    # Setting up the screen (for turtle)
    screen = turtle.Screen()
    turtle.speed(0)
    screen.setup(600, 600)

    # Run 50 iteration of Gradient Descent search
    for i in range(0, number_iterations):
        citylocations = gd_iteration(citylocations)

    # Pause turtle until clicked
    screen.exitonclick()


def get_arguments():
    global number_cities
    global number_iterations
    global rng_seed

    # If no command line arguments supplied then
    # default and return function
    if len(sys.argv) == 1:
        print(f"Usage: {sys.argv[0]} <number of cities> <number of "
              "iterations> <rng seed>")
        print(f"Defaulting to:\nnumber of cities={number_cities},"
              f"\nnumber of iterations={number_iterations},"
              f"\nrng seed={rng_seed}")
        return

    # If they have been supplied but cause an error
    # (are incorrect) print help and exit
    try:
        number_cities = int(sys.argv[1])
        number_iterations = int(sys.argv[2])
        rng_seed = int(sys.argv[3])
    except:
        try:
            number_cities = int(sys.argv[1])
            number_iterations = int(sys.argv[2])
        except:
            raise SystemExit(f"Usage: {sys.argv[0]}"
                             "<number of cities> <number of iterations>"
                             "<rng seed (optional)>")


def gd_iteration(citylocations):
    # Create a list of candidates
    candidates = create_candidates(citylocations)

    # Get a sorted list of candidates
    # lowest (i.e. best) first
    scored_candidates = sorted(candidates, key=objective_function)

    # The best scored candidate is our choice
    best_candidate = scored_candidates[0]

    # Draw its path, and return it
    drawpath(best_candidate)
    return best_candidate


def objective_function(candidate):
    sum = 0

    for i in range(0, number_cities):
        if i == number_cities - 1:
            sum = sum + euclidean_distance(candidate[-1], candidate[0])
        else:
            sum = sum + euclidean_distance(candidate[i], candidate[i+1])

    return sum


def euclidean_distance(pointa, pointb):
    return math.sqrt(
        math.pow(pointa[0] - pointb[0], 2) +
        math.pow(pointa[1] - pointb[1], 2))


def create_candidates(citylocations):
    candidates = []

    for i in range(0, number_cities):

        candidate = []

        if i == number_cities - 1:
            candidate.append(citylocations[-1])
            candidate.extend(citylocations[1:-1])
            candidate.append(citylocations[0])
        else:
            for j in range(0, number_cities):
                if not j == i and not j == i + 1:
                    candidate.append(citylocations[j])
                elif j == i and not i == number_cities - 1:
                    candidate.append(citylocations[i+1])
                    candidate.append(citylocations[i])

            if i == number_cities - 1:
                candidate.append(citylocations[0])

        candidates.append(candidate)

    return candidates


def drawpath(cities):
    turtle.clear()
    drawdistance(cities)
    turtle.penup()

    for city in cities:
        turtle.goto(city)
        turtle.pendown()

    turtle.goto(cities[0])


def drawdistance(cities):
    turtle.penup()
    turtle.goto(-290, -285)
    distance = objective_function(cities)
    turtle.write(f'Distance: {distance}', move=False,
                 align='left', font=('Arial', 8, 'normal'))


if __name__ == '__main__':
    main()
