// @ts-check
const { openiap } = require("@openiap/nodeapi");
const fs = require("fs");
const mongo = require("mongodb");
const tf = require("@tensorflow/tfjs-node");
const ExcelJS = require('exceljs');
var asciichart = require ('asciichart')

// import * as ExcelJS from 'exceljs';
// import * as tf from '@tensorflow/tfjs-node';
    // mongodb://openflow:0EXxzPh0gQLwMFbA2mWfFF3s6Q4TwC6T@openflow-mongodb-0.demo.openiap.io:443?replicaSet=rs0&tls=true
    // var client = new openiap();
    // client.onConnected = onConnected
    // await client.connect();

/** 
 * @param {openiap} client
 * **/
async function onConnected(client) {
}
// const cli = new mongo.MongoClient("mongodb://openflow:0EXxzPh0gQLwMFbA2mWfFF3s6Q4TwC6T@openflow-mongodb-0.demo.openiap.io:443?replicaSet=rs0&tls=true");
const cli = new mongo.MongoClient("mongodb://127.0.0.1:27017");
const db = cli.db("demo2");


const layers = 6; // 50;
const layerunits = 108; // 50;

const epochs = 210;
const batchesPerEpoch = 16;
const years = 10;
async function getModel(restart = false) {
    let model;
    if (restart == true) {
        model = tf.sequential();
        // model.add(tf.layers.flatten({ inputShape: [9, featureNames.length] }));
        model.add(tf.layers.lstm({ units: layerunits, returnSequences: true, inputShape: [9, featureNames.length] }));
        model.add(tf.layers.lstm({ units: layerunits }));
        // for (let i = 0; i < layers; i++) {
        //     // slowly decreate units from layerunits to featureNames.length
        //     let units = Math.floor(layerunits - (layerunits - featureNames.length) * (i / layers));

        //     // model.add(tf.layers.dense({ units, activation: "relu" })); 
        //     model.add(tf.layers.dense({ units: layerunits * 2, activation: "relu" })); 
        //     // sigmoid - create posebilities for each output
        //     // softmax - create posebility with only one outcome
        //     // relu - 
        // }
        model.add(tf.layers.dense({ units: featureNames.length }));
    } else {
        model = await tf.loadLayersModel('file://./cvr-model/model.json');
    }

    // const optimizer = tf.train.adam(0.001);
    const optimizer = tf.train.adam();
    // const optimizer = tf.train.sgd(0.001);
    // const optimizer = tf.train.adamax(0.001);
    // const optimizer = tf.train.rmsprop(0.005);
    model.compile({ 
        optimizer, // adam, sgd, adagrad, adadelta, adamax, rmsprop
        loss: 'meanSquaredError', 
        // meanSquaredError, categoricalCrossentropy, sparseCategoricalCrossentropy, binaryCrossentropy
        metrics: ['accuracy', 'mse'] // mse accuracy
    }); 
    // categoricalCrossentropy - output layer has a softmax activation and the labels are one-hot encoded.
    // meanSquaredError - generally used for regression tasks

    // model.layers.forEach((layer, index) => {
    //     if(index == 0) {
    //         // @ts-ignore
    //         console.log("input", layer.input.shape);
    //     }
    //     process.stdout.write(layer.name + " ");
    //     layer.outputShape.forEach(shape => (shape != null ? console.log(shape) : null));
    // });

    return model;
}
let loss = [];
let acc = [];
let mse = [];
let subloss = [];
let subacc = [];
let submse = [];

async function train_batched(restart) {
    var config1 = {
        height: 20,
        colors: [
            asciichart.red,
            asciichart.green,
            asciichart.white,
            asciichart.lightblue
        ]
    }
    const maxLength = 120
    var config2 = {
        height: 10,
        colors: [
            asciichart.green,
            asciichart.cyan,
        ]
    }
    let model = await getModel(restart);
    // @ts-ignore
    const ds = tf.data.generator(batchGenerator);
    await model.fitDataset(ds, { epochs, batchesPerEpoch, verbose: 0,
        validationData: ds,
        validationBatchSize: batchesPerEpoch,
        validationBatches: 5,
        callbacks: {
        onBatchBegin(batch, logs) {
        },
        onBatchEnd(batch, logs) {
            // @ts-ignore
            subloss.push(logs.loss);
            // @ts-ignore
            subacc.push(logs.acc);
            // @ts-ignore
            submse.push(logs.mse);
            if (subloss.length >= maxLength) subloss.shift ()
            if (subacc.length >= maxLength) subacc.shift ()
            if (submse.length >= maxLength) submse.shift ()
            try {
                console.clear();
                console.log (asciichart.plot ([subloss, subacc, loss, acc], config1));

                // @ts-ignore
                process.stdout.write (`\x1b[31m loss \x1b[0m${logs.loss.toString().padEnd(20, " ")}`)
                // @ts-ignore
                process.stdout.write (`\x1b[32m acc \x1b[0m${logs.acc.toString().padEnd(20, " ")}`)
                process.stdout.write (`\x1b[37m LOSS \x1b[0m${loss[loss.length - 1].toString().padEnd(20, " ")}`)
                process.stdout.write (`\x1b[92m ACC \x1b[0m${acc[acc.length - 1].toString().padEnd(20, " ")}`)
                process.stdout.write (`\x1b[92m MSE \x1b[0m${mse[mse.length - 1].toString().padEnd(20, " ")}`)
                // console.clear();
                // console.log (asciichart.plot ([subloss], config1));
                // console.log (asciichart.plot ([subacc, submse], config2));
            } catch (error) {
            }
        },
        onEpochEnd(epoch, logs) {
            // subloss = [];
            // subacc = [];
            // submse = [];
            // @ts-ignore
            loss.push(logs.loss);
            // @ts-ignore
            acc.push(logs.acc);
            // @ts-ignore
            mse.push(logs.mse);
            if (loss.length >= maxLength) loss.shift ()
            if (acc.length >= maxLength) acc.shift ()
            if (mse.length >= maxLength) mse.shift ()
            try {
                // console.clear();
                // console.log (asciichart.plot ([loss], config1));
                // console.log (asciichart.plot ([acc, mse], config2));
            } catch (error) {
            }            
        }
    } } );

    const saveResult = await model.save('file://./cvr-model');
    // console.log('Model saved:', saveResult);
}
const featureNames = ['calculatedEBIT','calculatedEBITDA','depreciationAmortization','employeeExpenses','externalExpenses','grossProfitLoss',
'otherOperatingExpenses','otherOperatingIncome','profitLoss','profitLossBeforeTax','profitLossFromOperatingActivities','Egenkapital'];
const headers = ['cvr', 'enhedsNummer', ...featureNames];
function imputeNullValues(data, defaultValue = 0) {
    return data.map(value => {
        if (Array.isArray(value)) {
            // Recursive call for nested arrays
            return imputeNullValues(value, defaultValue);
        }
        return value === null ? defaultValue : value;
    });
}

function flatternCompanyData(companyData) {
    companyData.forEach(company => {
        company.financial.forEach((yearData, index) => {
            const res = company.financial[index];
            if(res.report == null) return;
            var _data = Object.assign({ name: res.start }, res.report.incomeStatement);
            _data.contributedCapital = res.report.balance?.liabilitiesAndEquity?.equity?.contributedCapital
            _data.retainedEarnings = res.report.balance?.liabilitiesAndEquity?.equity?.retainedEarnings
            _data.shorttermDebtToBanks = res.report.balance?.liabilitiesOtherThanProvisions?.shorttermLiabilities?.shorttermDebtToBanks
            _data.shorttermTaxPayables = res.report.balance?.liabilitiesOtherThanProvisions?.shorttermLiabilities?.shorttermTaxPayables
            if (_data.contributedCapital == null) _data.contributedCapital = 0;
            if (_data.retainedEarnings == null) _data.retainedEarnings = 0;
            _data.Egenkapital = _data.contributedCapital + _data.retainedEarnings;
            var keys = Object.keys(_data);
            for (var j = 0; j < keys.length; j++) {
              var key = keys[j];
              if (_data[key] == null || _data[key] == 0) {
                delete _data[key];
              }
            }
            if(res.End == null) {
                res.End = new Date(res.start);
                res.End.setFullYear(res.End.getFullYear() + 1);
            }
            _data.End = res.End;
            featureNames.map((feature, index) => {
                if(_data[feature] == null) {
                    _data[feature] = 0; 
                }
            });
            company.financial[index] = _data;
        });

        let financials = company.financial;
        let count = financials.length;
        if (financials.length < 10) {
            let missing = 10 - financials.length;
            for (var y = 0; y < missing; y++) {
                const End = new Date(financials[0].End);
                const name = financials[0].End.toISOString().substring(0, 10);
                End.setDate(End.getDate() - 365);
                var obj = {name};
                featureNames.map(name => obj[name] = 0);
                obj.End = End;
                financials.unshift(obj);             
                // financials.unshift(featureNames.map(name => 0));
            }
        } else if (financials.length > 10) {
            // financials.sort((a, b) => a.End - b.End);
            // grab the last 10 years
            financials = financials.slice(financials.length - 10, financials.length);            
        }
        company.financial = financials;
        if(financials.length != 10) {
            debugger;
        }
    });
}
/**
 * 
 * @param data {any[]} data 
 * @returns {any} {x: number[][][], y: number[][],}
 */
function normalizeData(data) {
    // Find min and max for each feature
    let minValues = new Array(featureNames.length).fill(Infinity);
    let maxValues = new Array(featureNames.length).fill(-Infinity);
    
    // flatternCompanyData(data);

    // Loop over data to find min and max values
    data.forEach(company => {
        company.financial.forEach(yearData => {
            featureNames.forEach((feature, index) => {
                if (yearData[feature] < minValues[index]) minValues[index] = yearData[feature];
                if (yearData[feature] > maxValues[index]) maxValues[index] = yearData[feature];
            });
        });
    });

    // Normalize data
    data = data.map(company => {
        return company.financial.map(yearData => {
            return featureNames.map((feature, index) => {
                var res = 2 * ((yearData[feature] - minValues[index]) / (maxValues[index] - minValues[index])) - 1;
                if( Number.isNaN(res)) {
                    res = 0;
                }
                return res;
            });
        });
    });
    /**
     * @type {number[][][]}
     */
    var xres = [];
    /**
     * @type {number[][]}
     */
    var yres = [];
    for (var i = 0; i < data.length; i++) {
        let financials = data[i];
        if (financials.length < 10) {
            let missing = 10 - financials.length;
            for (var y = 0; y < missing; y++) {
                financials.unshift(featureNames.map(name => 0));
            }
        } else if (financials.length > 10) {
            financials = financials.slice(0, 10);
        }
        var last = financials[financials.length - 1];
        financials = financials.slice(0, financials.length - 1);
        xres.push(financials);
        yres.push(last);

    }
    xres = xres.map(companyData => companyData.map(yearData => imputeNullValues(yearData)));
    yres = yres.map(yearData => imputeNullValues(yearData));
 
    return {x: xres, y: yres, minValues, maxValues};
}
function denormalizeData(normalizedData, minValues, maxValues) {
    return normalizedData.map((companyData, idx) => {
        var result = {}
        companyData.map((value, index) => {
            result[featureNames[index]] = (value + 1) * (maxValues[index] - minValues[index]) / 2 + minValues[index];
            // return (value + 1) * (maxValues[index] - minValues[index]) / 2 + minValues[index];
        });
        return result;
    });
}
async function dataGenerator_old() {
    try {
        const numberOfBatches = batchesPerEpoch * epochs;
        let batch = 0;
        const today = new Date()
        const lastyear = new Date(today.getFullYear() - 1, today.getMonth(), today.getDate());
        const theearyears = new Date(today.getFullYear() - 3, today.getMonth(), today.getDate());
        const begin = new Date(today.getFullYear() - years, today.getMonth(), today.getDate());
        const cursor = db.collection("cvr").find({"stiftelsesDato": { $lte: theearyears }, "ophoersDato": {"$exists": false}, "virksomhedsformkort": "APS"}  )
            .project({financial: 1, "cvr": 1, "enhedsNummer": 1});
        while (batch < numberOfBatches) {
            try {
                console.log("");
                console.log("Begin - Batch " + batch + " of " + numberOfBatches);
                const dataBatch = [];
                const cvrData = [];
                do {
                    const row = await cursor.next();
                    if(row != null && row.cvr != null) {
                        row["financial"] = await db.collection("cvrfinancial").find({"enhedsNummer": row.enhedsNummer} ).project({"report": 1, "End": 1, "start": 1, }).toArray();
                        if(row["financial"].length > 1) {
                            dataBatch.push(row);
                        }
                    }
                } while (dataBatch.length < batchesPerEpoch);
                flatternCompanyData(dataBatch);
                for(let i = 0; i < dataBatch.length; i++) {
                    const row = dataBatch[i];
                    row["financial"].map(x=> {
                        var cvr = {...x};
                        cvr.cvr = row.cvr;
                        cvr.enhedsNummer = row.enhedsNummer;
                        cvrData.push(cvr);
                    });
                }
                const workbook = new ExcelJS.Workbook();
                let worksheet;
                if(fs.existsSync('financial.xlsx')) {
                    await workbook.xlsx.readFile('financial.xlsx');
                    fs.unlinkSync('financial.xlsx');
                }
                worksheet = workbook.getWorksheet("financial");
                if(worksheet == null) {
                    worksheet = workbook.addWorksheet("financial");
                }
                let columns = [];
                for (var i = 0; i < cvrData.length; i++) {
                    var keys = Object.keys(cvrData[i]);
                    for (var k = 0; k < keys.length; k++) {
                    if (columns.indexOf(keys[k]) == -1) {
                        columns.push(keys[k]);
                    }
                    }
                }
                worksheet.columns = columns.map(x => ({ header: x, key: x }));

                // var lastRow = worksheet?.lastRow;
                // for (var i = 0; i < cvrData.length; i++) {
                //     var r = worksheet?.addRow(cvrData[i]);
                // }
                worksheet?.addRows(cvrData);
                // console.log("Writing " + cvrData.length + " rows to excel with " + worksheet?.rowCount + " rows");
                await workbook.xlsx.writeFile("financial.xlsx");
                console.log("Return batch " + batch + " of " + numberOfBatches + " with " + dataBatch.length + " companies (" + worksheet?.rowCount + " total rows)");
                const {x, y} = normalizeData(dataBatch);
                var xs = tf.tensor3d(x);
                var ys = tf.tensor2d(y);

                // @ts-ignore
                batch++;
            } catch (error) {
                console.error(error);
            }
        }
    } catch (error) {
        console.error(error);
    }
}
/**
 * Async generator function to yield batches of data from the database.
 * @returns {AsyncGenerator<tf.Tensor, void, undefined>}
 */
async function* dataGenerator() {
    try {
        const numberOfBatches = batchesPerEpoch * epochs;
        let batch = 0;
        const today = new Date()
        const lastyear = new Date(today.getFullYear() - 1, today.getMonth(), today.getDate());
        const theearyears = new Date(today.getFullYear() - 3, today.getMonth(), today.getDate());
        const begin = new Date(today.getFullYear() - years, today.getMonth(), today.getDate());
        const cursor = db.collection("cvrfinancial").find() // .find({"stiftelsesDato": { $lte: theearyears }, "ophoersDato": {"$exists": false}, "virksomhedsformkort": "APS"}  )
            .project({financial: 1, "cvr": 1, "enhedsNummer": 1, "report": 1, "End": 1, "start": 1}).sort({"enhedsNummer": 1});
        while (batch < numberOfBatches) {
            try {
                console.log("");
                console.log("Begin - Batch " + batch + " of " + numberOfBatches);
                const dataBatch = [];
                const cvrData = [];
                let data = [];
                let lastcvr = null;
                do {
                    const row = await cursor.next();
                    if(row == null || lastcvr != row.cvr) {
                        lastcvr = row?.cvr;
                        if(data.length > 0) {
                            dataBatch.push({
                                financial: data,
                                enhedsNummer: data[0]?.enhedsNummer,
                                cvr: data[0]?.cvr,
                            });
                        }
                        data = [];
                    }
                    data.push(row);
                } while (dataBatch.length < batchesPerEpoch);
                flatternCompanyData(dataBatch);
                for(let i = 0; i < dataBatch.length; i++) {
                    const row = dataBatch[i];
                    row["financial"].map(x=> {
                        var cvr = {...x};
                        cvr.cvr = row.cvr;
                        cvr.enhedsNummer = row.enhedsNummer;
                        cvrData.push(cvr);
                    });
                }
                console.log("Return batch " + batch + " of " + numberOfBatches + " with " + dataBatch.length + " companies");
                const {x, y} = normalizeData(dataBatch);
                var xs = tf.tensor3d(x);
                var ys = tf.tensor2d(y);
                console.log("Shape of x before tensor conversion:", xs.shape);
                console.log("Shape of y before tensor conversion:", ys.shape);

                // @ts-ignore
                yield { xs, ys };
                batch++;
            } catch (error) {
                console.error(error);
            }
        }
    } catch (error) {
        console.error(error);
    }
}
var csvWriter = require('csv-write-stream');
async function savedata(filename, row) {
    let writer;
    if (!fs.existsSync(filename))
        writer = csvWriter({ headers: headers });
    else
        writer = csvWriter({ sendHeaders: false });
    writer.pipe(fs.createWriteStream(filename, { flags: 'a' }));
    writer.write({
        row
    });
    writer.end();
}
async function dumpdata() {
    let writer;
    if (!fs.existsSync("financial.csv")) {
        writer = csvWriter({ headers: headers });
    } else {
        writer = csvWriter({ sendHeaders: false, headers: headers });
    }        
    writer.pipe(fs.createWriteStream("financial.csv", { flags: 'a' }));
    try {
        let batch = 0;
        const cursor = db.collection("cvrfinancial").find()
            .project({financial: 1, "cvr": 1, "enhedsNummer": 1, "report": 1, "End": 1, "start": 1}).sort({"enhedsNummer": 1});
        const count = await db.collection("cvrfinancial").estimatedDocumentCount();
        console.log("Begin - Batch " + batch + " of " + count);
        while (await cursor.hasNext()) {
            try {
                console.log("Begin - Batch " + batch + " of " + Math.floor(count/ batchesPerEpoch));
                const dataBatch = [];
                let data = [];
                let lastcvr = null;
                do {
                    const row = await cursor.next();
                    if(row == null || lastcvr != row.cvr) {
                        lastcvr = row?.cvr;
                        if(data.length > 0) {
                            dataBatch.push({
                                financial: data,
                                enhedsNummer: data[0]?.enhedsNummer,
                                cvr: data[0]?.cvr,
                            });
                        }
                        data = [];
                    }
                    data.push(row);
                } while (dataBatch.length < batchesPerEpoch);
                flatternCompanyData(dataBatch);
                for(let i = 0; i < dataBatch.length; i++) {
                    const row = dataBatch[i];
                    row["financial"].map(x=> {
                        var cvr = {...x};
                        cvr.cvr = row.cvr;
                        cvr.enhedsNummer = row.enhedsNummer;
                        writer.write(cvr);
                    });
                    
                }
                batch++;
            } catch (error) {
                console.error(error);
            }
        }
    } catch (error) {
        console.error(error);
    } finally {
        writer.end();
    }
}

const parse_csv = require('parse-csv-stream');

async function* getRows2() {
    const readStream = fs.createReadStream('./financial.csv', 'utf8');
    const options = {
        // delimiter: ',',
        // wrapper: '"',
        // newlineSeparator: '\r\n'
    };
    const parser = new parse_csv(options);

    const rowQueue = [];
    let rowsProcessed = 0;

    // Wrap the parsing logic in a Promise
    const parsingPromise = new Promise((resolve, reject) => {
        parser.on('data', (row) => {
            var array = JSON.parse(row);
            if(array.length < 1) return;
            rowQueue.push(array[0]);

            if (rowQueue.length >= 10) {
                rowsProcessed += rowQueue.length;
                const batch = [...rowQueue];
                rowQueue.length = 0; // Clear the queue
                resolve(batch);
            }
        });

        parser.on('end', () => {
            if (rowQueue.length > 0) {
                rowsProcessed += rowQueue.length;
                resolve(rowQueue);
            }
        });

        parser.on('error', (err) => {
            reject(err);
        });

        readStream.pipe(parser);
    });

    while (true) {
        const batch = await parsingPromise;
        if (batch.length === 0) {
            break;
        }
        yield batch;
    }
}

const parse = require('csv-parser');
async function* getRows() {
    const batchSize = 10;
    let currentBatch = [];
    
    try {
      const stream = fs.createReadStream('financial.csv').pipe(parse(headers));
      
      for await (const row of stream) {
        if(row.cvr == null || row.cvr == "cvr") continue;
        currentBatch.push(row);
        
        if (currentBatch.length === batchSize) {
          yield currentBatch;
          currentBatch = [];
        }
      }
      
      if (currentBatch.length > 0) {
        yield currentBatch;
      }
    } catch (error) {
      console.error(error);
    }
  }

  
  async function train(restart) {
    let model = await getModel(restart);

    const rowsGenerator = getRows();
    let index = 0;
    for await (const _rows of rowsGenerator) {
        var rows = [];
        _rows.map(row => {
            var arr = [];
            featureNames.map((feature, index) => {
                arr.push(parseInt(row[feature]));
            });
            rows.push(arr);
        });
        let x = rows.splice(0, rows.length - 1);
        const y = rows.splice(rows.length - 1, 1);
        console.log(x);
        console.log(y);
        
        // x = x.filter(row => !row.every(val => val === 0));
        const _xs = tf.tensor2d(x);
        const xmin = _xs.min(0);
        const xmax = _xs.max(0);
        // Calculate range and add a small epsilon to avoid division by zero
        const epsilon = 1e-7;  // A small constant
        const range = xmax.sub(xmin).add(epsilon);
        // Normalize: (value - min) / (max - min + epsilon)
        const xs = _xs.sub(xmin).div(range).reshape([1, 9, 12]);

        const _ys = tf.tensor1d(y[0]);
        const ymin = _ys.min(0);
        const ymax = _ys.max(0);
        const yRange = ymax.sub(ymin).add(epsilon);
        const ys = _ys.sub(ymin).div(yRange).reshape([1, featureNames.length]);
        xs.print();
        ys.print();
        
        // Train the model
        var res = await model.fit(xs, ys, { epochs: 25, verbose: 0 });
        // console.log("Loss after Epoch " + res.epoch + " : " + res.history.loss[0])
        // console.log(res.history.loss[0])
        index++;
        // @ts-ignore
        const avgloss = res.history.loss.reduce((a, b) => a + b, 0) / res.history.loss.length;
        // @ts-ignore
        const avgacc = res.history.acc.reduce((a, b) => a + b, 0) / res.history.loss.length;
        // @ts-ignore
        const avgmse = res.history.mse.reduce((a, b) => a + b, 0) / res.history.loss.length;
        if(index % 100 == 0) {
            console.log(avgloss, avgacc, " *", index, "of", 5000)
            const saveResult = await model.save('file://./cvr-model');
        } else if(index % 1 == 0) {
            console.log(avgloss, avgacc, "  ", index, "of", 5000)
        }
        
    }
    const saveResult = await model.save('file://./cvr-model');
    // console.log('Model saved:', saveResult);
}



/**
 * return {any}
 */
async function* batchGenerator() {
    const rowsGenerator = getRows();
    let batchX = []; // Array to accumulate batch data for xs
    let batchY = []; // Array to accumulate batch data for ys

    for await (const _rows of rowsGenerator) {
        var rows = [];
        _rows.map(row => {
            var arr = [];
            featureNames.map(feature => {
                arr.push(parseInt(row[feature]));
            });
            rows.push(arr);
        });

        // Split the data into x and y
        const x = rows.splice(0, rows.length - 1);
        const y = rows.splice(rows.length - 1, 1);

        // Normalize x
        const _xs = tf.tensor2d(x);
        const xmin = _xs.min(0);
        const xmax = _xs.max(0);
        const epsilon = 1e-7; // A small constant
        const range = xmax.sub(xmin).add(epsilon);
        const xs = _xs.sub(xmin).div(range);

        // Normalize y
        const _ys = tf.tensor1d(y[0]);
        const ymin = _ys.min(0);
        const ymax = _ys.max(0);
        const yRange = ymax.sub(ymin).add(epsilon);
        const ys = _ys.sub(ymin).div(yRange);

        // Accumulate batch data
        batchX.push(xs.arraySync());
        batchY.push(ys.arraySync());

        // Check if the batch is complete
        if (batchX.length === batchesPerEpoch) {
            // @ts-ignore
            let batchedXs = tf.tensor3d(batchX, [batchesPerEpoch, 9, featureNames.length]);
            // @ts-ignore
            let batchedYs = tf.tensor2d(batchY, [batchesPerEpoch, featureNames.length]);

            yield { xs: batchedXs, ys: batchedYs };

            // Reset batch data
            batchX = [];
            batchY = [];
        }
    }
}


async function predict(model, companyData) {
    flatternCompanyData([companyData]);
    // const x = companyData.financial.splice(0, companyData.financial.length - 1);
    const x = companyData.financial.slice(0, 9);
    x.map((row, index) => {
        var arr = [];
        featureNames.map(name => {
            arr.push(row[name]);
        });
        x[index] = arr;
    });

    const _xs = tf.tensor2d(x);
    const xmin = _xs.min(0);
    const xmax = _xs.max(0);
    const epsilon = 1e-7; // A small constant
    const range = xmax.sub(xmin).add(epsilon);
    const xs = _xs.sub(xmin).div(range);
    const batched_xs = xs.reshape([1, 9, 12]); 

    var xminarray = xmin.arraySync();
    var xmaxarray = xmax.arraySync();

    const predictions = model.predict(batched_xs);
    let predictionsArray;
    if (Array.isArray(predictions)) {
        // If predictions is an array of tensors, convert each tensor to an array
        predictionsArray = await Promise.all(predictions.map(tensor => tensor.array()));
    } else {
        // If predictions is a single tensor, convert it to an array
        predictionsArray = await predictions.array();
    }
    const denormalizedPredictions = denormalizeData(predictionsArray, xminarray, xmaxarray);
    denormalizedPredictions[0].End = new Date(companyData.financial[companyData.financial.length - 1].End);
    denormalizedPredictions[0].name = denormalizedPredictions[0].End.toISOString().substring(0, 10);
    denormalizedPredictions[0].End.setDate(denormalizedPredictions[0].End.getDate() + 365);
    companyData.financial.push(denormalizedPredictions[0]);
}
async function train2222(restart) {
    let model = await getModel(restart);

    const rowsGenerator = getRows();
    for await (const _rows of rowsGenerator) {
        var rows = [];
        _rows.map(row => {
            var arr = [];
            featureNames.map((feature, index) => {
                arr.push(parseInt(row[feature]));
            });
            rows.push(arr);
        });
        const x = rows.splice(0, rows.length - 1);
        const y = rows.splice(rows.length - 1, 1);
        
        const _xs = tf.tensor2d(x);
        const xmin = _xs.min(0);
        const xmax = _xs.max(0);
        const __xs = _xs.sub(xmin).div(xmax.sub(xmin));
        const xs = __xs.reshape([1, 9, 12]);


        const _ys = tf.tensor1d(y[0]);
        // const ymin = _ys.min(0);
        // const ymax = _ys.max(0);
        // const __ys = _ys.sub(ymin).div(ymax.sub(ymin));
        const ys = _ys.reshape([1, 12]);


        // console.log("Shape of x before tensor conversion:", xs.shape);
        // console.log("Shape of y before tensor conversion:", ys.shape);


        await model.fit(xs, ys, { epochs: 1 });
        const saveResult = await model.save('file://./cvr-model');
        // console.log('Model saved:', saveResult);
    }

    // // @ts-ignore
    // const ds = tf.data.generator(dataGenerator);
    // await model.fitDataset(ds, { epochs, batchesPerEpoch });
    // const saveResult = await model.save('file://./cvr-model');
    // console.log('Model saved:', saveResult);
}
async function predict_old(model, companyData) {

    const normalizedData = normalizeData([companyData]); 
    const testData = tf.tensor3d(normalizedData.x);

    const predictions = model.predict(testData);
    let predictionsArray;
    if (Array.isArray(predictions)) {
        // If predictions is an array of tensors, convert each tensor to an array
        predictionsArray = await Promise.all(predictions.map(tensor => tensor.array()));
    } else {
        // If predictions is a single tensor, convert it to an array
        predictionsArray = await predictions.array();
    }
    const denormalizedPredictions = denormalizeData(predictionsArray, normalizedData.minValues, normalizedData.maxValues);
    companyData.financial.push(denormalizedPredictions[0]);
}
function dumpDinancials(data) {
    var result = data;
    if(result.financial) result = result.financial;
    let Egenkapital = [];
    let profitLoss = [];
    result = result.map((value) => {
        var res = {};
        ["name", ...featureNamestest].map((feature, index) => {
            if(feature == "name") {
                res[feature] = value[feature];
            } else {
                res[feature] = parseInt(value[feature]);
                if(feature == "Egenkapital") {
                    Egenkapital.push(res[feature]);
                }
                if(feature == "profitLoss") {
                    profitLoss.push(res[feature]);
                }
            }            
        });
        return res;
    });
    // console.table(result);
    var config = {
        height: 15,
        colors: [
            asciichart.green,
            asciichart.red,
            asciichart.white,
            asciichart.lightblue
        ]
    }
    console.clear();
    console.log (asciichart.plot ([Egenkapital, profitLoss], config));
    process.stdout.write (`\x1b[32m Egenkapital \x1b[0m${Egenkapital[Egenkapital.length -1 ].toString().padEnd(20, " ")}`)
    // change color to red
    process.stdout.write (`\x1b[31m profitLoss \x1b[0m${profitLoss[profitLoss.length - 1].toString().padEnd(20, " ")}`)
    console.log("");

}
async function testModel(cvrNumber) {
    // 1. Fetch the company's data
    /**
     * @type {any}
     */
    const companyData = await db.collection("cvr").findOne({ cvr: cvrNumber }, { projection: { financial: 1, "cvr": 1, "enhedsNummer": 1 } });
    companyData["financial"] = await db.collection("cvrfinancial").find({"enhedsNummer": companyData.enhedsNummer} ).project({"report": 1, "End": 1, "start": 1, }).toArray();

    // companyData.financial.pop();
    flatternCompanyData([companyData]);
    // dumpDinancials(companyData);
    const model = await getModel(false);

    let financials = companyData.financial;

    for(var i = 0; i < 30; i++) {
        await predict(model, companyData);
        financials.push(companyData.financial[companyData.financial.length - 1]);
        dumpDinancials(financials);
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    return companyData.financial;
}

const featureNamestest = ['employeeExpenses','grossProfitLoss','profitLoss','profitLossBeforeTax','Egenkapital'];

async function main() {
    // // // // // // // await dumpdata(); // Begin - Batch 8784 of 55664 - MongoCursorExhaustedError: Cursor is exhausted

    await train_batched(false);

    // await train(true);
    // var result = await testModel(39461455);
    // var result = await testModel(33152973); // b2bpresales
    // var result = await testModel(40400230); // openiap
    var result = await testModel(38725963); // velsmurt
    // var result = await testModel(38729365); // zenamic
    // 30359666

    

    cli.close();

}
main();