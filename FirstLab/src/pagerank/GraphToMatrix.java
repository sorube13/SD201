package pagerank;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

/*
 * Write the map and reduce function. To test your code run PublicTests.java. 
 * On the site submit a zip archive of your src folder. 
 * Try also the release tests after your submission. You have 3 trials per hour for the release tests. 
 * A correct implementation will get the same number of points for both public and release tests.
 * Please take the time to understand the settings for a job, in the next lab your will need to configure it by yourself. 
 */

public class GraphToMatrix {

	static class Map extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
			String line = value.toString();
			
			String[] elem = line.split("\t");
			IntWritable key_value = new IntWritable(Integer.parseInt(elem[0]));
			context.write(key_value, new IntWritable(Integer.parseInt(elem[1])));
		
			//throw new UnsupportedOperationException("Implementation missing");	
		}
	}

	static class Reduce extends Reducer<IntWritable, IntWritable, NullWritable, Text> {

		protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
			int sum = 0;
			List<Integer> valArray = new ArrayList<Integer>();
			//int [] valArray1 = null;
			for (IntWritable val : values){
				valArray.add(Integer.parseInt(val.toString()));
				sum++;
			}
			for(int i=0; i<valArray.size(); i++){
				double M = 0;
				if (sum != 0){
					M = 1.0/sum;
				}
				context.write(null, new Text((valArray.get(i)).toString() + " " + key.toString() + " " + String.valueOf(M)));
			}
			
			//throw new UnsupportedOperationException("Implementation missing");	
		}
	} 

	public static void job(Configuration conf) throws IOException, ClassNotFoundException, InterruptedException {
		Job job = Job.getInstance(conf);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(IntWritable.class);

		job.setMapperClass(Map.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.setInputPaths(job, new Path(conf.get("processedGraphPath")));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("stochasticMatrixPath")));
		job.waitForCompletion(true);
	}

}
