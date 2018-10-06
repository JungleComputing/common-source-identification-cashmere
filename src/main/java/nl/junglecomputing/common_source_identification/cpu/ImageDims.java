/*
 * Copyright 2018 Vrije Universiteit Amsterdam, The Netherlands
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
