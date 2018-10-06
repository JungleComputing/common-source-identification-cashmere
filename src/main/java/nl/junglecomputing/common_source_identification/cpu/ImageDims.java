package nl.junglecomputing.common_source_identification.cpu;

import java.io.File;
import java.io.IOException;

import java.util.Objects;

import java.awt.image.BufferedImage;

// simple class to return the height and width of the images that we are
// correlating.
public class ImageDims {
    public final int height;
    public final int width;

    public ImageDims(File imageFile) throws IOException {
	BufferedImage image = FileToImageStage.readImage(imageFile);
	height = image.getHeight();
	width = image.getWidth();
    }

    ImageDims(int height, int width) {
	this.height = height;
	this.width = width;
    }

    @Override
    public boolean equals(Object other) {
	boolean result = false;
	if (other instanceof ImageDims) {
	    ImageDims that = (ImageDims) other;
	    result = that.canEqual(this) && that.height == this.height && that.width == this.width;
	}
	return result;
    }

    @Override
    public int hashCode() {
	return Objects.hash(height, width);
    }

    public boolean canEqual(Object other) {
	return (other instanceof ImageDims);
    }
}
