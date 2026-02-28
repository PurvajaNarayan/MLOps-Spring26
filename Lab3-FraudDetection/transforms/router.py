import apache_beam as beam

class RouteTransaction(beam.DoFn):
    def process(self, element):
        if element["fraud_predicted"]:
            yield beam.pvalue.TaggedOutput("flagged", element)
        else:
            yield beam.pvalue.TaggedOutput("clean", element)