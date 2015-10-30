package pagerank;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

/*
 * Write the map and reduce functions for each job. For the second job write also the reduce function for the combiner.
 * To test your code run PublicTests.java.
 * On the site submit a zip archive of your src folder.
 * Try also the release tests after your submission. You have 3 trials per hour for the release tests. 
 * A correct implementation will get the same number of points for both public and release tests.
 * Please take the time also to understand the settings for a job, in the next lab your will need to configure it by yourself. 
 */

public class MatrixVectorMult {

	static class FirstMap extends Mapper<LongWritable, Text, IntWritable, Text> {

		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
		{
			String line = value.toString();			
			String[] elem = line.split("\\s");
			
			if(elem.length == 3){
				IntWritable key_value = new IntWritable(Integer.parseInt(elem[1]));
				context.write(key_value, new Text(elem[0] + " " + elem[2]));
			}else if(elem.length == 2){
				IntWritable key_value = new IntWritable(Integer.parseInt(elem[0]));
				context.write(key_value, new Text(elem[1]));
			}

			//throw new UnsupportedOperationException("Implementation missing");	
		}
	}


	static class FirstReduce extends Reducer<IntWritable, Text, IntWritable, DoubleWritable> {

		protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException{

			HashMap<Integer,Double> valArray = new HashMap<>();
			double v=0;
			
			for (Text val : values){
				String[] str = val.toString().split("\\s");
				if (str.length != 1){
					valArray.put(Integer.parseInt(str[0]), Double.parseDouble(str[1]));
				}else{
					v = Double.parseDouble(str[0]);
				}
			}
			
			for( int k:valArray.keySet()){
				context.write(new IntWritable(k), new DoubleWritable(valArray.get(k)*v));
//			
			//throw new UnsupportedOperationException("Implementation missing");	
			}
		}
		}

	static class SecondMap extends Mapper<Text, Text, IntWritable, DoubleWritable> {

		protected void map(Text key, Text value, Context context) throws IOException, InterruptedException
		{

			context.write(new IntWritable(Integer.parseInt(key.toString())), new DoubleWritable(Double.parseDouble(value.toString())));	
			
			//throw new UnsupportedOperationException("Implementation missing");	

		}
	}

	static class CombinerForSecondMap extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {

		protected void reduce(IntWritable key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException{
			List<String> BArray = new ArrayList<String>();
			for (DoubleWritable val : values){
				BArray.add(val.toString());
			}
			double B = 0;
			for(int j=0; j<BArray.size(); j++){
				B += Double.parseDouble(BArray.get(j));
			}
			context.write(key, new DoubleWritable(B));
			
			//throw new UnsupportedOperationException("Implementation missing");	
		}
	}

	static class SecondReduce extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {

		protected void reduce(IntWritable key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException{

			List<String> BArray = new ArrayList<String>();
			for (DoubleWritable val : values){
				BArray.add(val.toString());
			}
			double B = 0;
			for(int j=0; j<BArray.size(); j++){
				B += Double.parseDouble(BArray.get(j));
			}
			context.write(key, new DoubleWritable(B));
			//throw new UnsupportedOperationException("Implementation missing");	
		}
	}

	public static void job(Configuration conf)
			throws IOException, ClassNotFoundException, InterruptedException {
		// First job
		Job job1 = Job.getInstance(conf);
		job1.setMapOutputKeyClass(IntWritable.class);
		job1.setMapOutputValueClass(Text.class);

		job1.setMapperClass(FirstMap.class);
		job1.setReducerClass(FirstReduce.class);

		job1.setInputFormatClass(TextInputFormat.class);
		job1.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.setInputPaths(job1, new Path[]{new Path(conf.get("initialVectorPath")), new Path(conf.get("inputMatrixPath"))});
		FileOutputFormat.setOutputPath(job1, new Path(conf.get("intermediaryResultPath")));

		job1.waitForCompletion(true);

		Job job2 = Job.getInstance(conf);
		job2.setMapOutputKeyClass(IntWritable.class);
		job2.setMapOutputValueClass(DoubleWritable.class);

		job2.setMapperClass(SecondMap.class);
		job2.setReducerClass(SecondReduce.class);

		/* If your implementation of the combiner passed the unit test, uncomment the following line*/
		//job2.setCombinerClass(CombinerForSecondMap.class);

		job2.setInputFormatClass(KeyValueTextInputFormat.class);
		job2.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.setInputPaths(job2, new Path(conf.get("intermediaryResultPath")));
		FileOutputFormat.setOutputPath(job2, new Path(conf.get("currentVectorPath")));

		job2.waitForCompletion(true);

		FileUtils.deleteQuietly(new File(conf.get("intermediaryResultPath")));
	}

	}



