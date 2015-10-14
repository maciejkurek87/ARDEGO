    #######################
    ## Abstract Methods  ##
    #######################
    
import operator
import random

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import pdb

def optimize(function, design_space, GEN=10000):
    toolbox = copy(base.Toolbox())
    D = len(design_space)
    
    best = None
    max_speed = 0.2
    smin = [-1.0 * max_speed *
            (dimSetting['max'] - dimSetting['min'])
            for dimSetting in design_space]
    smax = [max_speed *
            (dimSetting['max'] - dimSetting['min'])
            for dimSetting in design_space]

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    ### we got to add specific names, otherwise the classes are going to be visible for all
    ### modules which use deap...
    
    creator.create(str('Particle', list, fitness=creator.FitnessMax),
                   smin=smin, smax=smax,
                   speed=[uniform(smi, sma) for sma, smi in zip(smax,
                                                                smin)],
                   pmin=[dimSetting['max'] for dimSetting in design_space],
                   pmax=[dimSetting['min'] for dimSetting in design_space],
                   model=False, best=None, code=None)

    toolbox.register('particle', generate, designSpace=design_space)
    toolbox.register('filter_particles', filterParticles,
                          designSpace=design_space)
    toolbox.register('filter_particle', filterParticle,
                          designSpace=design_space)
    toolbox.register('population', tools.initRepeat,
                          list, toolbox.particle)
    toolbox.register('update', updateParticle, 
                          conf=self.get_configuration(),
                          designSpace=design_space)
    toolbox.register('evaluate', self.fitness_function)
    
    ### create all the neccesary functions
    pop = toolbox.population(n=D*10)
    filter_population(pop)
    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
        filter_population(pop)
    return best

def filterParticles(particles, designSpace):
    for particle in particles:
        self.filterParticle(particle, designSpace)
        
def filterParticle(p, designSpace):
    p.pmin = [dimSetting['min'] for dimSetting in designSpace]
    p.pmax = [dimSetting['max'] for dimSetting in designSpace]

    for i, val in enumerate(p):
        #dithering
        if designSpace[i]['type'] == 'discrete':
            if uniform(0.0, 1.0) < (p[i] - floor(p[i])):
                p[i] = ceil(p[i])  # + designSpace[i]['step']
            else:
                p[i] = floor(p[i])

        #dont allow particles to take the same value
        p[i] = minimum(p.pmax[i], p[i])
        p[i] = maximum(p.pmin[i], p[i])

def generate(designSpace):
    particle = [uniform(dimSetting['min'], dimSetting['max'])
                for dimSetting
                in designSpace]
    particle = create_particle(particle)
    return particle

# update the position of the particles
# should change this part in order to change the leader election strategy
def updateParticle(part, generation, max_iter, designSpace):
    fraction = generation / max_iter

    u1 = [uniform(0, 2.0) for _ in range(len(part))]
    u2 = [uniform(0, 2.0) for _ in range(len(part))]
    
    ##########   this part particulately, leader election for every particle
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, self.get_best(), part))
    weight = 1.0
    
    weightVector = [weight] * len(part.speed)
    part.speed = map(operator.add,
                     map(operator.mul, part.speed, weightVector),
                     map(operator.add, v_u1, v_u2))


    for i, speed in enumerate(part.speed):
        maxVel = (1 - pow(fraction, 2.0)) * part.smax[i]
        if speed < -maxVel:
            part.speed[i] = -maxVel
        elif speed > maxVel:
            part.speed[i] = maxVel

    part[:] = map(operator.add, part, part.speed)

def filter_population(sel, pop):
    toolbox.filter_particles(pop)

    