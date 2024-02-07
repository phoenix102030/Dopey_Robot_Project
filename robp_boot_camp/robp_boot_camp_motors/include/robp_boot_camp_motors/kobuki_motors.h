#ifndef KOBUKI_MOTORS_H
#define KOBUKI_MOTORS_H

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <vector>

class KobukiMotors {
 public:
  KobukiMotors();

  virtual ~KobukiMotors();

  //    updates kobuki motors
  //    input: size 2 vector of pwm signals (signal range between -255 and 255)
  //           pwm[0] --> left wheel
  //           pwm[1] --> right wheel
  //    output: size 2 (0: left wheel, 1: right wheel)
  //            angular_velocities: angular velocities of each wheel [rad]
  //            abs_encoders: absolute encoder values for each wheel
  //            diff_encoders: differential encoder values for each wheel
  void update(const std::vector<int> &pwm,
              std::vector<double> &angular_velocities,
              std::vector<int> &abs_encoders, std::vector<int> &diff_encoders);

 private:
  // variables used for generating noise
  boost::mt19937 eng_;
  boost::normal_distribution<double> noise_distribution_;
  boost::variate_generator<boost::mt19937, boost::normal_distribution<double> >
      generator_;

  // stored absolute encoder values
  std::vector<int> abs_encoders_;
};

#endif  // KOBUKI_MOTORS_H