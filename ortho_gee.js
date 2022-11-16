//FUNCTION PACKAGE TO GENERATE ORTHOGONAL ANGLES FOR EACH RIVER XS



//INPUTS
var arcrw  = ee.FeatureCollection("users/jflores/koshi_R_centerlines"); //vector river centerlines


//FUNCTIONS
var HitOrMiss = function(image, se1, se2) {
  // perform hitOrMiss transform
  var e1 = image.reduceNeighborhood(ee.Reducer.min(), se1);
  var e2 = image.not().reduceNeighborhood(ee.Reducer.min(), se2);

  return(e1.and(e2));
};
var SplitKernel = function(kernel, value) {
  // recalculate the kernel according to the given foreground value
  var result = [];
  for(var r = 0; r < kernel.length; r++) {
    var row = [];
    for(var c = 0; c < kernel.length; c++) {
      row.push(kernel[r][c] == value ? 1 : 0);
    }
    result.push(row);
  }

  return(result);
};

var ExtractEndpoints = function(CL1px) {
  // calculate end points in the one pixel centerline

  var se1w = [[0, 0, 0],
            [2, 1, 2],
            [2, 2, 2]];

  var se11 = ee.Kernel.fixed(3, 3, SplitKernel(se1w, 1));
  var se12 = ee.Kernel.fixed(3, 3, SplitKernel(se1w, 2));

  var result = CL1px;
  // the for loop removes the identified endpoints from the imput image
  for(var i=0; i<4; i++) { // rotate kernels

    result = result.subtract(HitOrMiss(result, se11, se12));

    se11 = se11.rotate(1);
    se12 = se12.rotate(1);
  }

  var endpoints = CL1px.subtract(result);
  return(endpoints);
};
var ExtractCorners = function(CL1px) {
  // calculate corner points in the one pixel centerline

  var se1w = [[2, 2, 0],
            [2, 1, 1],
            [0, 1, 0]];

  var se11 = ee.Kernel.fixed(3, 3, SplitKernel(se1w, 1));
  var se12 = ee.Kernel.fixed(3, 3, SplitKernel(se1w, 2));

  var result = CL1px;
  // the for loop removes the identified corners from the imput image
  for(var i=0; i<4; i++) { // rotate kernels

    result = result.subtract(HitOrMiss(result, se11, se12));

    se11 = se11.rotate(1);
    se12 = se12.rotate(1);
  }

  var cornerPoints = CL1px.subtract(result);
  return(cornerPoints);
};
exports.CleanCenterline = function(cl1px, maxBranchLengthToRemove, rmCorners) {
  //*** clean the 1px centerline:	1. remove branches 2. remove corners to insure 1px width (optional)


  var nearbyPoints = cl1px.mask(cl1px).reduceNeighborhood({
    reducer: ee.Reducer.count(),
    kernel: ee.Kernel.circle(1.5),
    skipMasked: true});

  var endsByNeighbors = nearbyPoints.lte(2);

  var joints = nearbyPoints.gte(4);

  var costMap = cl1px.mask(cl1px).updateMask(joints.not()).cumulativeCost({
    source: endsByNeighbors.mask(endsByNeighbors),
    maxDistance: maxBranchLengthToRemove,
    geodeticDistance: false});

  var branchMask = costMap.gte(0).unmask(0);
  var cl1Cleaned = cl1px.updateMask(branchMask.not()); // mask short branches;
  var ends = ExtractEndpoints(cl1Cleaned);
  cl1Cleaned = cl1Cleaned.updateMask(ends.not());

  if (rmCorners) {
    var corners = ExtractCorners(cl1Cleaned);
    cl1Cleaned = cl1Cleaned.updateMask(corners.not());
  }
  return(cl1Cleaned);
};

exports.CalculateAngle = function(clCleaned) {
  // calculate the orthogonal direction of each pixel of the centerline

  var w3 = (ee.Kernel.fixed(9, 9, [
  [135.0, 126.9, 116.6, 104.0, 90.0, 76.0, 63.4, 53.1, 45.0],
  [143.1, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 36.9],
  [153.4, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 26.6],
  [166.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 14.0],
  [180.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 1e-5],
  [194.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 346.0],
  [206.6, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 333.4],
  [216.9, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 323.1],
  [225.0, 233.1,  243.4,  256.0,  270.0,  284.0,  296.6,  306.9, 315.0]]));

  var combinedReducer = ee.Reducer.sum().combine(ee.Reducer.count(), null, true);

  var clAngle = (clCleaned.mask(clCleaned)
      .rename(['clCleaned'])
      .reduceNeighborhood({
      reducer: combinedReducer,
      kernel: w3,
      inputWeight: 'kernel',
      skipMasked: true}));

  // ## mask calculating when there are more than two inputs into the angle calculation
  var clAngleNorm = (clAngle
      .select('clCleaned_sum')
      .divide(clAngle.select('clCleaned_count'))
      .mask(clAngle.select('clCleaned_count').gt(2).not()));

  // ## if only one input into the angle calculation, rotate it by 90 degrees to get the orthogonal
  clAngleNorm = (clAngleNorm
      .where(clAngle.select('clCleaned_count').eq(1), clAngleNorm.add(ee.Image(90))));

  return(clAngleNorm.rename(['orthDegree']));
}; 

exports.getgeometry = function(feature){
  var geo = feature.geometry().coordinates();
  var id = feature.get('COMID');
  var width = feature.get('mwth_mean'); //width_m
  var xsDist_m = feature.get('distance'); //distance upstream
  var xsid = feature.get('Feature Index'); //xs ID
  return(ee.Feature(null,{
    'lat':geo.get(1),
    'lon':geo.get(0),
    'ID': id,
    'width':width,
    'xsDist_m':xsDist_m,
    'xsID':xsid
  }));
};

var getUTMProj = function(lon) {
  // only works for northerm hemisphere
  // see https://apollomapping.com/blog/gtm-finding-a-utm-zone-number-easily
  var utmCode = ee.Number(lon).add(180).divide(6).ceil().int();
  return(ee.String('EPSG:326').cat(utmCode.format('%02d')));
};

var CreateXsectionGen = function(width) {
  return(function(f) {
    var utmPrj = getUTMProj(f.get('lon'));
    //var utmPrj = 'EPSG:32612';
    var xy = ee.Geometry.Point([f.get('lon'), f.get('lat')]).transform(utmPrj, 1).coordinates();
    var x = ee.Number(xy.get(0));
    var y = ee.Number(xy.get(1));
    var orthAngle = ee.Number(f.get('angle')).divide(180).multiply(Math.PI);
    var dx = orthAngle.cos().multiply(width).divide(2); 
    var dy = orthAngle.sin().multiply(width).divide(2);
    
    var geo = ee.Geometry.LineString([x.add(dx), y.add(dy), x.subtract(dx), y.subtract(dy)], utmPrj, false);
    return(f.setGeometry(geo));
  });
};

exports.CalOrthogonalAngle = function(mypoint) {
  var f = ee.Feature(mypoint);
  var crs = getUTMProj(f.get('lon'));//.aside(print);
 // var crs = 'EPSG:32612'
  
  var width = ee.Number(f.get('width'));
  var CalcXSection = CreateXsectionGen(width.multiply(1.5));
  var aoi = f.geometry().buffer(1500);//.aside(Map.centerObject)
  var cl = ee.Image(0).toByte().paint(arcrw, 1).clip(aoi).reproject(crs, null, 3);//.aside(Map.addLayer, {min: 0, max: 1});
  var angle = exports.CalculateAngle(cl).reproject(crs, null, 3); //.aside(Map.addLayer, {min: 0, max: 360})
  var bound = angle.geometry();
  var scale = 3;
  
  var clPoints = (angle.rename(['angle'])
  .addBands(ee.Image.pixelCoordinates(crs))
  .addBands(ee.Image.pixelLonLat().rename(['lon', 'lat']))
  .addBands(ee.Image(width).rename('width'))
  .sample({
      region: bound,
      scale: scale,
      projection: null,
      factor: 1,
      dropNulls: true
  }));

  var getDistance = function(f){
    var point = ee.Geometry.Point([f.get('lon'),f.get('lat')]);
    var center = ee.Geometry.Point([mypoint.get('lon'),mypoint.get('lat')]);
    var geo = f.geometry().coordinates();
    var dis = (ee.Feature(ee.Geometry.Point([f.get('lon'), f.get('lat')]), {
        //'MLength': f.get('MLength'),
                 'crs': crs,
                 'lat': f.get('lat'),
                 'lon': f.get('lon'),
                 'OrthoAngle': f.get('angle'),
      //           'xc': f.get('x'),
        //         'yc': f.get('y'),
                 'distance': point.distance(center),
                 'width': mypoint.get('width'),
                 'xsID': mypoint.get('xsID'),
                 'xsDist_m': mypoint.get('xsDist_m'),
          //       'st_lat': ee.List(geo.get(0)).get(1),
            //     'st_lon': ee.List(geo.get(0)).get(0),
              //   'end_lat': ee.List(geo.get(1)).get(1),
              //   'end_lon': ee.List(geo.get(1)).get(0),
                 'ID': mypoint.get('ID')                 
      }));
      return(dis);
  };

  var collection = clPoints.map(CalcXSection);
  var mp = collection.map(getDistance).sort('distance').first();

  return(mp);
};


//HYDROGRAPHY RIVER XS
var rp = ee.FeatureCollection('users/jflores/HMA/merit20_all'); //river cross-section points
print(rp.size(), rp.first())

//CALCULATE ORTHGONAL ANGLES
var lonlat = ee.FeatureCollection(rp.map(exports.getgeometry));
var sites = ee.FeatureCollection(lonlat.map(function(i) { 
  return(ee.Feature(ee.Geometry.Point([i.get('lon'), i.get('lat')]), {
    'lon':i.get('lon'),
    'lat':i.get('lat'),
    'ID':i.get('ID'),
    'width':i.get('width'),
    'xsDist_m': i.get('xsDist_m'),
    'xsID': i.get('xsID')
  }));
}));
print('Number of XS: ',sites.size());
//print('Number of Images: ',imageCollection.size());

var rp = sites.map(exports.CalOrthogonalAngle); 
rp = ee.FeatureCollection(rp);

//SAVE TO ASSET
Export.table.toAsset({
  collection: rp,
  description:'merit20_all',
  assetId: 'HMA/merit20_all',
});