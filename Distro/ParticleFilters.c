/*
  Particle filters implementation for a simple robot.

  Your goal here is to implement a simple particle filter
  algorithm for robot localization.

  A map file in .ppm format is read from disk, the map
  contains empty spaces and walls.

  A simple robot is randomly placed on this map, and
  your task is to write a particle filter algorithm to
  allow the robot to determine its location with
  high certainty.

  You must complete all sections marked

  TO DO:

  NOTE: 2 keyboard controls are provided:

  q -> quit (exit program)
  r -> reset particle set during simulation

  Written by F.J.E. for CSC C85, May 2012. Updated Aug. 15, 2014

  This robot model inspired by Sebastian Thrun's
  model in CS373.
*/

#include "ParticleFilters.h"

/**********************************************************
 GLOBAL DATA
**********************************************************/
unsigned char *map;        // Input map
unsigned char *map_b;        // Temporary frame
struct particle *robot;        // Robot
struct particle *list, *list_new;        // Particle list
struct particle *list_cumu;        // Particle list
int sx, sy;            // Size of the map image
char name[1024];        // Name of the map
int n_particles;        // Number of particles
int windowID;                // Glut window ID (for display)
int Win[2];                    // window (x,y) size
int RESETflag;            // RESET particles

double uniform_random(void);

void CalCumulative(void);

double cal_avg(struct particle *pt);
double cal_sigma(struct particle *pt);
double Guass_Function(double x,double sigma);
double m[16];

/**********************************************************
 PROGRAM CODE
**********************************************************/
int main(int argc, char *argv[]) {
    /*
      Main function. Usage for this program:

      ParticleFilters map_name n_particles

      Where:
       map_name is the name of a .ppm file containing the map. The map
                should be BLACK on empty (free) space, and coloured
                wherever there are obstacles or walls. Anythin not
                black is an obstacle.

       n_particles is the number of particles to simulate in [100, 50000]

      Main loads the map image, initializes a robot at a random location
       in the map, and sets up the OpenGL stuff before entering the
       filtering loop.
    */

    if (argc != 3) {
        fprintf(stderr, "Wrong number of parameters. Usage: ParticleFilters map_name n_particles.\n");
        exit(0);
    }

    strcpy(&name[0], argv[1]);
    n_particles = atoi(argv[2]);

    if (n_particles < 100 || n_particles > 50000) {
        fprintf(stderr, "Number of particles must be in [100, 50000]\n");
        exit(0);
    }

    fprintf(stderr, "Reading input map\n");
    map = readPPMimage(name, &sx, &sy);
    if (map == NULL) {
        fprintf(stderr, "Unable to open input map, or not a .ppm file\n");
        exit(0);
    }

    // Allocate memory for the temporary frame
    fprintf(stderr, "Allocating temp. frame\n");
    map_b = (unsigned char *) calloc(sx * sy * 3, sizeof(unsigned char));
    if (map_b == NULL) {
        fprintf(stderr, "Out of memory allocating image data\n");
        free(map);
        exit(0);
    }

    srand48((long) time(NULL));        // Initialize random generator from timer
    fprintf(stderr, "Init robot...\n");
    robot = initRobot(map, sx, sy);
    if (robot == NULL) {
        fprintf(stderr, "Unable to initialize robot.\n");
        free(map);
        free(map_b);
        exit(0);
    }
    sonar_measurement(robot, map, sx, sy);    // Initial measurements...
    for(int j=0;j<16;j++) {
        m[j] = robot->measureD[j];
    }

    // Initialize particles at random locations
    fprintf(stderr, "Init particles...\n");
    list = NULL;
    initParticles();

    // Done, set up OpenGL and call particle filter loop
    fprintf(stderr, "Entering main loop...\n");
    Win[0] = 800;
    Win[1] = 800;
    glutInit(&argc, argv);
    initGlut(argv[0]);
    glutMainLoop();

    // This point is NEVER reached... memory leaks must be resolved by OpenGL main loop
    deleteList(list);
    deleteList(list_cumu);
    deleteList(list_new);
    exit(0);

}

void initParticles(void) {
    /*
      This function creates and returns a linked list of particles
      initialized with random locations (not over obstacles or walls)
      and random orientations.

      There is a utility function to help you find whether a particle
      is on top of a wall.

      Use the global pointer 'list' to keep trak of the *HEAD* of the
      linked list.

      Probabilities should be uniform for the initial set.
    */

    //list = NULL;

    /***************************************************************
    // TO DO: Complete this function to generate an initially random
    //        list of particles.
    ***************************************************************/
    struct particle *p, *pt, *p_c;
    list = NULL;
    list_new = NULL;
    list_cumu = NULL;
    //set partical list
    list = new particle;
    //set other list for resampling the partical from the list
    list_new = new particle;
    //cumu_list is for save the CDF of list prob.
    list_cumu = new particle;

    list =initRobot(map,sx,sy);
    ground_truth(list, map, sx, sy);
    list->prob = (float) (1) / ((float) n_particles);

    list->next = NULL;
    list_new->next = NULL;
    list_cumu->next = NULL;
    p = list;
    pt = list_new;
    p_c = list_cumu;
    int i = 1;
    while (i < n_particles) {
        particle *list_t = new particle;
        particle *list_new_t = new particle;
        particle *list_new_c = new particle;
        // set each particle randomly and away from the wall
        list_t = initRobot(map, sx, sy);
        ground_truth(list_t, map, sx, sy);
        list_t->prob = (float) (1) / ((float) n_particles);

        list_t->next = NULL;
        list_new_t->next = NULL;
        list_new_c->next = NULL;


        p->next = list_t;
        pt->next = list_new_t;
        p_c->next = list_new_c;

        p = list_t;
        pt = list_new_t;
        p_c = list_new_c;

        i += 1;
        printf("part= %i prob= %lf\n", i, p->prob);
    }
}

void computeLikelihood(struct particle *p, struct particle *rob, double noise_sigma) {
    /*
      This function computes the likelihood of a particle p given the sensor
      readings from robot 'robot'.

      Both particles and robot have a measurements array 'measureD' with 16
      entries corresponding to 16 sonar slices.

      The likelihood of a particle depends on how closely its measureD
      values match the values the robot is 'observing' around it. Of
      course, the measureD values for a particle come from the input
      map directly, and are not produced by a simulated sonar.

      Assume each sonar-slice measurement is independent, and if

      error_i = (p->measureD[i])-(sonar->measureD[i])

      is the error for slice i, the probability of observing such an
      error is given by a Gaussian distribution with sigma=20 (the
      noise standard deviation hardcoded into the robot's sonar).

      You may want to check your numbers are not all going to zero...

      This function updates the likelihood for the particle in
      p->prob
    */

    /****************************************************************
    // TO DO: Complete this function to calculate the particle's
    //        likelihood given the robot's measurements
    ****************************************************************/

    double error_i;
    double error_prob = 0;
    // compute particles' 16 sonar slices, add all the error probability together.
    for (int i = 0; i < 16; i++) {
        error_i = (p->measureD[i]) - (rob->measureD[i]);
        error_prob = error_prob + log(GaussEval(error_i, noise_sigma));
    }
    // multiply the past probability
    p->prob = log(p->prob) + error_prob;
    // change log in to number
    p->prob = exp(p->prob);

}

void ParticleFilterLoop(void) {
    /*
       Main loop of the particle filter
    */

    // OpenGL variables. Do not remove
    unsigned char *tmp;
    GLuint texture;
    static int first_frame = 1;
    double max;
    struct particle *p, *pmax;
    char line[1024];
    // Add any local variables you need right below.

    if (!first_frame) {
        // Step 1 - Move all particles a given distance forward (this will be in
        //          whatever direction the particle is currently looking).
        //          To avoid at this point the problem of 'navigating' the
        //          environment, any particle whose motion would result
        //          in hitting a wall should be bounced off into a random
        //          direction.
        //          Once the particle has been moved, we call ground_truth(p)
        //          for the particle. This tells us what we would
        //          expect to sense if the robot were actually at the particle's
        //          location.
        //
        //          Don't forget to move the robot the same distance!
        //

        /******************************************************************
        // TO DO: Complete Step 1 and test it
        //        You should see a moving robot and sonar figure with
        //        a set of moving particles.
        ******************************************************************/
        double  distance = 2 ,theta_c=180,sigma_sample=100;
        //seting the robat moving dirction
        p=list;
        while (p!=NULL) {
            //1 Move all particles a given distance forward
            //set every partical moving direct follow the robat moving dirction
            //p->theta = robot->theta+uniform_random()*20;
            // if the particles hit the wall, then turn 180 degree
            if (hit(p, map, sx, sy)) {
                p->theta = p->theta + theta_c + uniform_random() * 20.0;
                move(p, distance);
            } else {
                move(p, distance);
            }
            //2 Once the particle has been moved, we call ground_truth(p)
            ground_truth(p, map, sx, sy);
            p = p->next;
        }
        //3 Don't forget to move the robot the same distance!
        // if the robot hit the wall, then turn 180 degree.
        if(hit(robot,map,sx,sy)) {
            robot->theta = robot->theta + theta_c + uniform_random() * 20.0;
            move(robot, distance);
        }else {
            move(robot, distance);
        }

        // Step 2 - The robot makes a measurement - use the sonar
        sonar_measurement(robot, map, sx, sy);

        // Step 3 - Compute the likelihood for particles based on the sensor
        //          measurement. See 'computeLikelihood()' and call it for
        //          each particle. Once you have a likelihood for every
        //          particle, turn it into a probability by ensuring that
        //          the sum of the likelihoods for all particles is 1.

        /*******************************************************************
        // TO DO: Complete Step 3 and test it
        //        You should see the brightness of particles change
        //        depending on how well they agree with the robot's
        //        sonar measurements. If all goes well, particles
        //        that agree with the robot's position/direction
        //        should be brightest.
        *******************************************************************/

        p = list;
        double sum_prob = 0;

        while (p != NULL) {
            computeLikelihood(p, robot, sigma_sample);
            //for nomalization
            sum_prob = sum_prob + p->prob;
            p = p->next;
        }

        //nomalize
        p = list;
        while (p != NULL) {
            p->prob = p->prob / sum_prob;
            p = p->next;
        }

        // Step 4 - Resample particle set based on the probabilities. The goal
        //          of this is to obtain a particle set that better reflect our
        //          current belief on the location and direction of motion
        //          for the robot. Particles that have higher probability will
        //          be selected more often than those with lower probability.
        //
        //          To do this: Create a separate (new) list of particles,
        //                      for each of 'n_particles' new particles,
        //                      randomly choose a particle from  the current
        //                      set with probability given by the particle
        //                      probabilities computed in Step 3.
        //                      Initialize the new particles (x,y,theta)
        //                      from the selected particle.
        //                      Note that particles in the current set that
        //                      have high probability may end up being
        //                      copied multiple times.
        //
        //                      Once you have a new list of particles, replace
        //                      the current list with the new one. Be sure
        //                      to release the memory for the current list
        //                      before you lose that pointer!
        //

        /*******************************************************************
        // TO DO: Complete and test Step 4
        //        You should see most particles disappear except for
        //        those that agree with the robot's measurements.
        //        Hopefully the largest cluster will be on and around
        //        the robot's actual location/direction.
        *******************************************************************/
        //calculation Cumulative probility of particla
        CalCumulative();
        double urandum = 0;
        struct particle *p, *pt, *ptt, *p_new;
        struct particle *list_t, *pindex;
        p = list;
        p_new = list_new;
        pindex = list;
        while (p != NULL) {
            //calculation the uniform RV
            urandum = uniform_random();
            pt = list;
            ptt = list_cumu;//cdf
            //find partical
            while (pt != NULL) {
                //find the well partical
                if (ptt->prob >= urandum) {
                    pindex = pt;//select ok.
                    pt = NULL;
                } else {
                    pt = pt->next;
                    ptt = ptt->next;
                }
            }
            //copy the well partical into p_new.
            p_new->x = pindex->x;
            p_new->y = pindex->y;
            p_new->theta = pindex->theta;
            p_new->prob = pindex->prob;

            p = p->next;
            p_new = p_new->next;
        }
        //update new partical
        //copy back the partical from newlist to list.
        p = list;
        p_new = list_new;
        sum_prob = 0;//for nomalization
        while (p != NULL) {
            p->x = p_new->x;
            p->y = p_new->y;
            p->theta = p_new->theta;
            p->prob = p_new->prob;
            sum_prob = sum_prob + p->prob;
            p = p->next;
            p_new = p_new->next;
        }


        p = list;

        while (p != NULL) {
            p->prob = p->prob / sum_prob;
            p = p->next;
        }

    }  // End if (!first_frame)

    /***************************************************
     OpenGL stuff
     You DO NOT need to read code below here. It only
     takes care of updating the screen.
    ***************************************************/
    if (RESETflag)    // If user pressed r, reset particles
    {
        deleteList(list);
        list = NULL;
        initParticles();
        RESETflag = 0;
    }
    renderFrame(map, map_b, sx, sy, robot, list);

    // Clear the screen and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);

    glGenTextures(1, &texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sx, sy, 0, GL_RGB, GL_UNSIGNED_BYTE, map_b);

    // Draw box bounding the viewing area
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(0.0, 100.0, 0.0);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(800.0, 100.0, 0.0);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(800.0, 700.0, 0.0);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(0.0, 700.0, 0.0);
    glEnd();

    p = list;
    max = 0;
    while (p != NULL) {
        if (p->prob > max) {
            max = p->prob;
            pmax = p;
        }
        p = p->next;
    }

    if (!first_frame) {
        sprintf(&line[0], "X=%3.2f, Y=%3.2f, th=%3.2f, EstX=%3.2f, EstY=%3.2f, Est_th=%3.2f, Error=%f", robot->x,
                robot->y, robot->theta, \
           pmax->x, pmax->y, pmax->theta,
                sqrt(((robot->x - pmax->x) * (robot->x - pmax->x)) + ((robot->y - pmax->y) * (robot->y - pmax->y))));
        glColor3f(1.0, 1.0, 1.0);
        glRasterPos2i(5, 22);
        for (int i = 0; i < strlen(&line[0]); i++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, (int) line[i]);
    }

    // Make sure all OpenGL commands are executed
    glFlush();
    // Swap buffers to enable smooth animation
    glutSwapBuffers();

    glDeleteTextures(1, &texture);

    // Tell glut window to update ls itself
    glutSetWindow(windowID);
    glutPostRedisplay();

    if (first_frame) {
        fprintf(stderr, "All set! press enter to start\n");
        gets(&line[0]);
        first_frame = 0;
    }
}

/*********************************************************************
 OpenGL and display stuff follows, you do not need to read code
 below this line.
*********************************************************************/
// Initialize glut and create a window with the specified caption
void initGlut(char *winName) {
    // Set video mode: double-buffered, color, depth-buffered
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    // Create window
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(Win[0], Win[1]);
    windowID = glutCreateWindow(winName);

    // Setup callback functions to handle window-related events.
    // In particular, OpenGL has to be informed of which functions
    // to call when the image needs to be refreshed, and when the
    // image window is being resized.
    glutReshapeFunc(WindowReshape);   // Call WindowReshape whenever window resized
    glutDisplayFunc(ParticleFilterLoop);   // Main display function is also the main loop
    glutKeyboardFunc(kbHandler);
}

void kbHandler(unsigned char key, int x, int y) {
    if (key == 'r') { RESETflag = 1; }
    if (key == 'q') {
        deleteList(list);
        free(map);
        free(map_b);
        exit(0);
    }
}

void WindowReshape(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();            // Initialize with identity matrix
    gluOrtho2D(0, 800, 800, 0);
    glViewport(0, 0, w, h);
    Win[0] = w;
    Win[1] = h;
}

double uniform_random(void) {
    return (double) rand() / (double) RAND_MAX;
}

//calculate the cumulative prob. Save to list_cumu
void CalCumulative() {
    struct particle *p, *pt;

    p = list->next;
    list_cumu->prob = list->prob;
    pt = list_cumu->next;
    double temp_prob;
    temp_prob = list_cumu->prob;
    while (p != NULL) {
        pt->prob = temp_prob + p->prob;
        temp_prob = pt->prob;

        p = p->next;
        pt = pt->next;
    }
}

//this function is to calculate the average prob.
double cal_avg(struct particle *pt)
{
    struct particle *p;
    p=pt;
    double temp_prob = 0;
    int i = 1;
    while (p!=NULL)
    {
        temp_prob = temp_prob + p->prob;

        i+=1;
        p=p->next;
    }
    return temp_prob/i;
}
//this function to calculate the sigma (SD) of prob
double cal_sigma(struct particle *pt)
{
    struct particle *p;
    double avg_prob = cal_avg(pt);
    p=pt;
    double temp_prob = 0;
    int i = 1;
    while (p!=NULL)
    {
        temp_prob = temp_prob + (p->prob - avg_prob)*(p->prob - avg_prob);

        i+=1;
        p=p->next;
    }
    return sqrt(temp_prob/i);
}
//this is Guass function
double Guass_Function(double x,double sigma){
    double prob;
    prob = -(x*x)/(2*sigma*sigma);
    prob = exp(prob);
    prob = prob/(sigma*sqrt(2*3.1415926));
    return prob;
}
