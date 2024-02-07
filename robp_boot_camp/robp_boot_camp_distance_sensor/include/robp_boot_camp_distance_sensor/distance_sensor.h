#ifndef DISTANCE_SENSOR_H
#define DISTANCE_SENSOR_H

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

class DistanceSensor {
 public:
  DistanceSensor();

  virtual ~DistanceSensor();

  // returns voltage measured by the sensor
  // for a given distance (measured in meters)
  double sample(double distance);

 private:
  // variables used for generating noise
  boost::mt19937 eng_;
  boost::normal_distribution<double> noise_distribution_;
  boost::variate_generator<boost::mt19937, boost::normal_distribution<double> >
      generator_;
};

#endif  // DISTANCE_SENSOR_H